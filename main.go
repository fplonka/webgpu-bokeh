// Package main implements a web server that generates background blur effects using
// the Depth Anything v2 model from Hugging Face for depth estimation.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	_ "image/png"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"os"
	"strings"
	"time"

	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"
)

const (
	// Hugging Face Space endpoint for the Depth Anything v2 model
	apiBaseURL   = "https://depth-anything-depth-anything-v2.hf.space"
	pollTimeout  = 30 * time.Second
	pollInterval = 2 * time.Second
	maxFileSize  = 10 << 20 // 10 MB

	// Index for grayscale depth map in the model's output array
	// (0: slider view, 1: grayscale, 2: 16-bit raw)
	depthMapIndex = 1
)

// Types for interacting with the Hugging Face Spaces API.
// The API uses a two-phase process:
// 1. Upload file and get event ID
// 2. Poll for results using event ID
type (
	uploadResponse []string

	eventIDResponse struct {
		EventID string `json:"event_id"`
	}

	fileData struct {
		Path string            `json:"path"`
		Meta map[string]string `json:"meta"`
	}

	apiRequest struct {
		Data []fileData `json:"data"`
	}
)

func init() {
	// Configure structured JSON logging for better observability
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))
}

// uploadToHF sends an image to the Hugging Face server using multipart form data.
// Returns the server-side path where the image was stored.
func uploadToHF(imageBytes []byte, filename string) (string, error) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	part, err := writer.CreateFormFile("files", filename)
	if err != nil {
		return "", fmt.Errorf("create form file: %w", err)
	}

	if _, err := part.Write(imageBytes); err != nil {
		return "", fmt.Errorf("write image bytes: %w", err)
	}
	writer.Close()

	req, err := http.NewRequest("POST", apiBaseURL+"/upload", &buf)
	if err != nil {
		return "", fmt.Errorf("create upload request: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("upload request failed: %w", err)
	}
	defer resp.Body.Close()

	var result uploadResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode upload response: %w", err)
	}

	if len(result) == 0 {
		return "", fmt.Errorf("empty upload response")
	}

	slog.Info("uploaded image", "path", result[0])
	return result[0], nil
}

// getEventID initiates depth map generation for an uploaded image.
// Returns an event ID used to poll for results.
func getEventID(uploadedPath string) (string, error) {
	reqData := apiRequest{
		Data: []fileData{{
			Path: uploadedPath,
			Meta: map[string]string{"_type": "gradio.FileData"},
		}},
	}

	reqBody, err := json.Marshal(reqData)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", apiBaseURL+"/call/on_submit", bytes.NewBuffer(reqBody))
	if err != nil {
		return "", fmt.Errorf("create API request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("API request failed: %w", err)
	}
	defer resp.Body.Close()

	var result eventIDResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode event ID response: %w", err)
	}

	slog.Info("got event ID", "id", result.EventID)
	return result.EventID, nil
}

// pollForResult waits for depth map generation to complete, with a 30s timeout.
// Returns normalized depth values (0-1) and image dimensions.
func pollForResult(eventID string) ([]float32, int, int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), pollTimeout)
	defer cancel()

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil, 0, 0, fmt.Errorf("polling timed out after %v", pollTimeout)
		case <-ticker.C:
			resp, err := http.Get(fmt.Sprintf("%s/call/on_submit/%s", apiBaseURL, eventID))
			if err != nil {
				return nil, 0, 0, fmt.Errorf("polling request failed: %w", err)
			}

			body, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				return nil, 0, 0, fmt.Errorf("read polling response: %w", err)
			}

			bodyStr := string(body)
			if bodyStr == "event: error\ndata: null\n\n" {
				return nil, 0, 0, fmt.Errorf("API returned error")
			}
			if bodyStr == "" || !strings.HasPrefix(bodyStr, "event: complete") {
				continue
			}

			// Extract and combine data lines
			var dataLines []string
			for _, line := range strings.Split(bodyStr, "\n") {
				if strings.HasPrefix(line, "data: ") {
					dataLines = append(dataLines, strings.TrimPrefix(line, "data: "))
				}
			}

			jsonStr := strings.Join(dataLines, "")
			var result []interface{}
			if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
				slog.Warn("failed to parse response JSON", "error", err)
				continue
			}

			// Get depth map URL and download it
			if depthMapURL, err := extractDepthMapURL(result); err == nil {
				return downloadDepthMap(depthMapURL)
			}
		}
	}
}

// extractDepthMapURL gets the depth map URL from the response
func extractDepthMapURL(result []interface{}) (string, error) {
	if len(result) <= depthMapIndex {
		return "", fmt.Errorf("response too short: %d elements", len(result))
	}

	depthMap, ok := result[depthMapIndex].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("invalid depth map data at index %d", depthMapIndex)
	}

	path, ok := depthMap["path"].(string)
	if !ok {
		return "", fmt.Errorf("no path in depth map data")
	}

	url := fmt.Sprintf("%s/file=%s", apiBaseURL, path)
	slog.Info("got depth map URL", "url", url)
	return url, nil
}

// downloadDepthMap retrieves and processes the depth map PNG.
// Converts grayscale PNG values to normalized floats (0-1) for WebGPU processing.
func downloadDepthMap(url string) ([]float32, int, int, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	// Read the PNG data
	img, err := png.Decode(resp.Body)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("decode PNG: %w", err)
	}

	bounds := img.Bounds()
	width := bounds.Max.X - bounds.Min.X
	height := bounds.Max.Y - bounds.Min.Y

	// Convert to grayscale values
	values := make([]float32, width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, _, _, _ := img.At(x, y).RGBA()
			// Convert from 0-65535 to 0-1 range
			values[y*width+x] = float32(r) / 65535.0
		}
	}

	slog.Info("processed depth map", "width", width, "height", height)
	return values, width, height, nil
}

// fetchDepthMap coordinates the full depth map generation process:
// upload → initiate → poll → download → process
func fetchDepthMap(imageBytes []byte) ([]float32, int, int, error) {
	uploadedPath, err := uploadToHF(imageBytes, "image.jpg")
	if err != nil {
		return nil, 0, 0, fmt.Errorf("upload failed: %w", err)
	}

	eventID, err := getEventID(uploadedPath)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("get event ID failed: %w", err)
	}

	values, width, height, err := pollForResult(eventID)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("polling failed: %w", err)
	}

	return values, width, height, nil
}

// handleUpload processes image uploads, generates depth maps via Hugging Face,
// and returns normalized depth values for WebGPU-based background blur.
func handleUpload(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse the multipart form data
	if err := r.ParseMultipartForm(maxFileSize); err != nil {
		slog.Error("failed to parse form", "error", err)
		http.Error(w, "Failed to parse form", http.StatusBadRequest)
		return
	}

	file, _, err := r.FormFile("image")
	if err != nil {
		slog.Error("failed to get file", "error", err)
		http.Error(w, "Failed to get file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Read the image
	fileBytes, err := io.ReadAll(file)
	if err != nil {
		slog.Error("failed to read file", "error", err)
		http.Error(w, "Failed to read file", http.StatusBadRequest)
		return
	}

	// Verify image format
	if _, format, err := image.Decode(bytes.NewReader(fileBytes)); err != nil {
		slog.Error("failed to decode image", "format", format, "error", err)
		http.Error(w, "Failed to decode image", http.StatusBadRequest)
		return
	} else {
		slog.Info("decoded image", "format", format)
	}

	// Fetch depth map from Hugging Face
	depthValues, width, height, err := fetchDepthMap(fileBytes)
	if err != nil {
		slog.Error("failed to fetch depth map", "error", err)
		http.Error(w, "Failed to generate depth map", http.StatusInternalServerError)
		return
	}

	// Set content type before writing response
	w.Header().Set("Content-Type", "application/json")

	response := map[string]interface{}{
		"depth_map": map[string]interface{}{
			"values": depthValues,
			"width":  width,
			"height": height,
		},
	}

	if err := json.NewEncoder(w).Encode(response); err != nil {
		slog.Error("failed to encode response", "error", err)
	}
}

func main() {
	// Serve WebGPU shaders and frontend from /static
	http.Handle("/", http.FileServer(http.Dir("static")))

	// REST endpoint for depth map generation
	http.HandleFunc("/api/upload", handleUpload)

	slog.Info("server starting", "addr", "http://localhost:8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		slog.Error("server failed", "error", err)
		os.Exit(1)
	}
}
