import os
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    if rc != 0:
        print(f"Command failed with exit code {rc}")
        sys.exit(rc)

def main():
    image_name = "vibe-reader-app"
    container_name = "vibe-reader-container"
    
    print("🚀 Building and Deploying Vibe Reader...")
    
    # Check if we are in the right directory
    if not os.path.exists("Dockerfile"):
        print("Error: Dockerfile not found. Please run this script from the project root.")
        sys.exit(1)

    # Build Docker Image
    print("\n📦 Building Docker Image...")
    # Using --network=host might help with pip download speed/issues in some environments
    run_command(f"docker build -t {image_name} .")
    
    # Run Docker Container
    print("\n🚢 Running Docker Container...")
    # Stop and remove existing container if it exists
    subprocess.run(f"docker stop {container_name}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(f"docker rm {container_name}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Run new container
    # Mapping port 8501 to 8501
    # Mounting models directory so we can use trained models without rebuilding image everytime (optional, but good for dev)
    # But for a "deployment" script usually we bake models in OR mount them.
    # The Dockerfile does 'COPY . .', so models are baked in if they exist at build time.
    # If users want to use latest models from host without rebuild, we can add a volume mount.
    
    cmd = f"docker run -d -p 8501:8501 --name {container_name} {image_name}"
    run_command(cmd)
    
    print("\n✅ Deployment Successful!")
    print(f"👉 Access the app at: http://localhost:8501")
    print(f"   (Container name: {container_name})")

if __name__ == "__main__":
    main()
