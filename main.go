// Package main implements a web server that generates background blur effects using
// the Depth Anything v2 model from Replicate for depth estimation.
package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	_ "image/png"
	"io"
	"log/slog"
	"net/http"
	"os"
	"time"

	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"
)

const (
	apiBaseURL   = "https://api.replicate.com/v1/predictions"
	modelVersion = "b239ea33cff32bb7abb5db39ffe9a09c14cbc2894331d1ef66fe096eed88ebd4"
	pollTimeout  = 30 * time.Second
	pollInterval = 2 * time.Second
	maxFileSize  = 10 << 20 // 10 MB
)

// Types for interacting with the Replicate API
type (
	replicateRequest struct {
		Version string `json:"version"`
		Input   struct {
			Image string `json:"image"`
		} `json:"input"`
	}

	replicateResponse struct {
		ID     string `json:"id"`
		Status string `json:"status"`
		Output struct {
			GreyDepth string `json:"grey_depth"`
		} `json:"output"`
		Error *string `json:"error"`
	}
)

func init() {
	// Configure structured JSON logging for better observability
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))

	// Ensure REPLICATE_API_TOKEN is set
	if os.Getenv("REPLICATE_API_TOKEN") == "" {
		slog.Error("REPLICATE_API_TOKEN environment variable not set")
		os.Exit(1)
	}
}

// callReplicate sends an image to the Replicate API and initiates depth map generation.
// Returns a prediction ID used to poll for results.
func callReplicate(imageBytes []byte) (string, error) {
	// Convert image to base64
	base64Image := base64.StdEncoding.EncodeToString(imageBytes)

	// Prepare request
	request := replicateRequest{
		Version: modelVersion,
		Input: struct {
			Image string `json:"image"`
		}{
			Image: "data:image/jpeg;base64," + base64Image,
		},
	}

	body, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", apiBaseURL, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	// Add required headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+os.Getenv("REPLICATE_API_TOKEN"))
	req.Header.Set("Prefer", "wait")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	var result replicateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	if result.Error != nil {
		return "", fmt.Errorf("api error: %s", *result.Error)
	}

	slog.Info("created prediction", "id", result.ID)
	return result.ID, nil
}

// pollForResult waits for depth map generation to complete, with a 30s timeout.
// Returns normalized depth values (0-1) and image dimensions.
func pollForResult(predictionID string) ([]float32, int, int, error) {
	slog.Info("starting to poll for result", "prediction_id", predictionID)
	endTime := time.Now().Add(pollTimeout)

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for time.Now().Before(endTime) {
		req, err := http.NewRequest("GET", apiBaseURL+"/"+predictionID, nil)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("create status request: %w", err)
		}

		req.Header.Set("Authorization", "Bearer "+os.Getenv("REPLICATE_API_TOKEN"))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("status request failed: %w", err)
		}
		defer resp.Body.Close()

		var result replicateResponse
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return nil, 0, 0, fmt.Errorf("decode status response: %w", err)
		}

		if result.Error != nil {
			return nil, 0, 0, fmt.Errorf("api error: %s", *result.Error)
		}

		switch result.Status {
		case "succeeded":
			slog.Info("prediction succeeded", "prediction_id", predictionID)
			return downloadDepthMap(result.Output.GreyDepth)
		case "failed":
			slog.Error("prediction failed", "prediction_id", predictionID)
			return nil, 0, 0, fmt.Errorf("prediction failed")
		default:
			slog.Debug("prediction in progress", "prediction_id", predictionID, "status", result.Status)
			<-ticker.C
		}
	}

	slog.Error("polling timed out", "prediction_id", predictionID, "timeout", pollTimeout)
	return nil, 0, 0, fmt.Errorf("timeout waiting for depth map")
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
// call replicate → poll → download → process
func fetchDepthMap(imageBytes []byte) ([]float32, int, int, error) {
	// Call Replicate API and get prediction ID
	predictionID, err := callReplicate(imageBytes)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("call replicate: %w", err)
	}

	// Poll for result and return depth map data
	return pollForResult(predictionID)
}

// handleUpload processes image uploads, generates depth maps via Replicate,
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

	// Fetch depth map from Replicate
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
