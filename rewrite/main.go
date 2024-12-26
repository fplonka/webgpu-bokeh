package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"  // Register GIF format
	_ "image/jpeg" // Register JPEG format
	_ "image/png"  // Register PNG format

	_ "golang.org/x/image/tiff" // Register TIFF format from x/image
	_ "golang.org/x/image/webp" // Register WebP format from x/image

	"io"
	"log"
	"mime/multipart"
	"net/http"
	"strings"
	"time"
)

const (
	// Index in the response array for different outputs:
	// 0: "Depth Map with Slider View" (Imageslider component)
	// 1: "Grayscale depth map" (File component)
	// 2: "16-bit raw output" (File component)
	DEPTH_MAP_INDEX = 1 // Change this to try different outputs
)

type HFRequest struct {
	Data []map[string]string `json:"data"`
}

type HFResponse struct {
	Done bool     `json:"done"`
	Data []string `json:"data"`
}

func uploadToHF(imageBytes []byte, filename string) (string, error) {
	// Create a buffer for the multipart form data
	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)

	// Create form file field
	part, err := writer.CreateFormFile("files", filename)
	if err != nil {
		return "", fmt.Errorf("failed to create form file: %v", err)
	}

	// Write the image bytes
	if _, err := part.Write(imageBytes); err != nil {
		return "", fmt.Errorf("failed to write image bytes: %v", err)
	}

	// Close the writer
	writer.Close()

	// Make the upload request
	req, err := http.NewRequest("POST", "https://depth-anything-depth-anything-v2.hf.space/upload", &requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to create upload request: %v", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to upload file: %v", err)
	}
	defer resp.Body.Close()

	// Read response
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read upload response: %v", err)
	}

	// Response is a JSON array with one string (the file path)
	var result []string
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return "", fmt.Errorf("failed to parse upload response: %v", err)
	}

	if len(result) == 0 {
		return "", fmt.Errorf("empty upload response")
	}

	return result[0], nil
}

func fetchDepthMap(imageBytes []byte) ([]byte, error) {
	// First upload the file
	uploadedPath, err := uploadToHF(imageBytes, "image.jpg")
	if err != nil {
		return nil, fmt.Errorf("failed to upload image: %v", err)
	}

	fmt.Printf("File uploaded successfully, got path: %s\n", uploadedPath)

	// Prepare the request to Hugging Face using the uploaded file path
	reqData := map[string]interface{}{
		"data": []interface{}{
			map[string]interface{}{
				"path":      uploadedPath,
				"orig_name": "image.jpg",
				"meta": map[string]string{
					"_type": "gradio.FileData",
				},
			},
		},
	}

	reqBody, err := json.Marshal(reqData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	fmt.Printf("Sending request body: %s\n", string(reqBody))

	// Make the initial POST request
	req, err := http.NewRequest("POST", "https://depth-anything-depth-anything-v2.hf.space/call/on_submit", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")

	// Send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make POST request: %v", err)
	}
	defer resp.Body.Close()

	// Read the response body
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %v", err)
	}
	fmt.Println("Response body:", string(bodyBytes))

	// Parse the event ID from the bytes we already read
	var result struct {
		EventID string `json:"event_id"`
	}
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	fmt.Printf("Got event ID: %s\n", result.EventID)

	// Poll for results
	for i := 0; i < 30; i++ { // Try for 30 seconds max
		fmt.Println("i is", i)
		resp, err := http.Get(fmt.Sprintf(
			"https://depth-anything-depth-anything-v2.hf.space/call/on_submit/%s",
			result.EventID,
		))
		if err != nil {
			return nil, fmt.Errorf("failed to make GET request: %v", err)
		}

		// Read and print the response body for polling requests
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			resp.Body.Close()
			return nil, fmt.Errorf("failed to read polling response body: %v", err)
		}
		resp.Body.Close()

		bodyStr := string(bodyBytes)
		fmt.Printf("Polling response body (raw):\n%s\n", bodyStr)

		// Check if it's an error response
		if bodyStr == "event: error\ndata: null\n\n" {
			return nil, fmt.Errorf("received error from API")
		}

		// Parse SSE format
		if bodyStr == "" {
			// Empty response, keep polling
			time.Sleep(time.Second)
			continue
		}

		// Check if it's a complete event
		if !strings.HasPrefix(bodyStr, "event: complete") {
			time.Sleep(time.Second)
			continue
		}

		// Extract the data part
		dataLines := []string{}
		for _, line := range strings.Split(bodyStr, "\n") {
			if strings.HasPrefix(line, "data: ") {
				dataLines = append(dataLines, strings.TrimPrefix(line, "data: "))
			}
		}

		// Combine all data lines into one JSON string
		jsonStr := strings.Join(dataLines, "")
		fmt.Printf("Combined JSON: %s\n", jsonStr)

		// Parse the combined JSON
		var result []interface{}
		if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
			fmt.Printf("Failed to parse combined JSON: %v\n", err)
			time.Sleep(time.Second)
			continue
		}

		// The depth map we want is at DEPTH_MAP_INDEX in the response array
		if len(result) <= DEPTH_MAP_INDEX {
			fmt.Printf("Response array too short, only got %d elements\n", len(result))
			continue
		}

		// Get the depth map data
		depthMap, ok := result[DEPTH_MAP_INDEX].(map[string]interface{})
		if !ok {
			fmt.Printf("Failed to parse depth map at index %d\n", DEPTH_MAP_INDEX)
			continue
		}

		// Get the depth map path
		path, ok := depthMap["path"].(string)
		if !ok {
			continue
		}

		fmt.Printf("Got depth map path from index %d: %s\n", DEPTH_MAP_INDEX, path)

		// Construct the correct URL for downloading
		downloadURL := fmt.Sprintf("https://depth-anything-depth-anything-v2.hf.space/file=%s", path)
		fmt.Printf("Downloading from URL: %s\n", downloadURL)

		// Download the depth map file
		resp, err = http.Get(downloadURL)
		if err != nil {
			return nil, fmt.Errorf("failed to download depth map: %v", err)
		}
		defer resp.Body.Close()

		return io.ReadAll(resp.Body)
	}

	return nil, fmt.Errorf("timeout waiting for depth map")
}

func main() {
	// Serve static files
	http.Handle("/", http.FileServer(http.Dir("static")))

	// Handle image upload and depth map generation
	http.HandleFunc("/api/upload", handleUpload)

	fmt.Println("Server running on http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func handleUpload(w http.ResponseWriter, r *http.Request) {
	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// Handle preflight requests
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse the multipart form data
	err := r.ParseMultipartForm(10 << 20) // 10 MB max
	if err != nil {
		fmt.Println("Failed to parse form", err)
		http.Error(w, "Failed to parse form", http.StatusBadRequest)
		return
	}

	file, _, err := r.FormFile("image")
	if err != nil {
		fmt.Println("Failed to get file", err)
		http.Error(w, "Failed to get file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Read the image
	fileBytes, err := io.ReadAll(file)
	if err != nil {
		fmt.Println("Failed to read file", err)
		http.Error(w, "Failed to read file", http.StatusBadRequest)
		return
	}

	// Verify image format
	img, format, err := image.Decode(bytes.NewReader(fileBytes))
	if err != nil {
		fmt.Printf("Failed to decode image (format: %s): %v\n", format, err)
		http.Error(w, "Failed to decode image", http.StatusBadRequest)
		return
	}
	fmt.Printf("Successfully decoded image of format: %s\n", format)

	// Fetch depth map from Hugging Face
	depthMapBytes, err := fetchDepthMap(fileBytes)
	if err != nil {
		fmt.Printf("Failed to fetch depth map: %v\n", err)
		http.Error(w, "Failed to generate depth map", http.StatusInternalServerError)
		return
	}

	// Set content type before writing response
	w.Header().Set("Content-Type", "application/json")

	response := map[string]interface{}{
		"depth_map": map[string]string{
			"data":   "data:image/png;base64," + base64.StdEncoding.EncodeToString(depthMapBytes),
			"width":  fmt.Sprint(img.Bounds().Max.X - img.Bounds().Min.X),
			"height": fmt.Sprint(img.Bounds().Max.Y - img.Bounds().Min.Y),
		},
	}

	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Error encoding response: %v", err)
	}
}
