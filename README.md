# good-bg-blur

web app for background blur using depth estimation and webgpu

## requirements

- go 1.21+
- tailwind cli
- browser with webgpu support
- replicate api token

## setup

```sh
# set api token
export REPLICATE_API_TOKEN=your_token_here

# start tailwind watcher
tailwindcss -i ./static/input.css -o ./static/output.css --watch

# run server
go run .
```

server runs on :8080 by default

## webgpu setup




