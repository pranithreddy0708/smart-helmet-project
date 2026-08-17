"""
Smart Helmet System — Deployment & Management Tool
Automates local deployment, Docker deployment, and testing workflows.

Usage:
    python deploy.py --mode local      (Start Live Web Server locally)
    python deploy.py --mode docker     (Build & Run with Docker Compose)
    python deploy.py --mode test       (Run system test suite)
"""

import sys
import subprocess
import argparse
import time

def run_cmd(cmd, cwd=None):
    print(f"🚀 Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        res = subprocess.run(cmd, cwd=cwd, shell=True if isinstance(cmd, str) else False)
        return res.returncode == 0
    except Exception as e:
        print(f"❌ Execution error: {e}")
        return False

def deploy_local(port=5050):
    print(f"\n🌐 Starting Smart Helmet Live Web Server on port {port}...")
    cmd = [sys.executable, "src/raspberry_pi/live_server.py", "--port", str(port)]
    run_cmd(cmd)

def deploy_docker():
    print("\n🐳 Deploying via Docker Compose...")
    if not run_cmd(["docker", "--version"]):
        print("❌ Docker is not installed or available in PATH.")
        return False
    
    success = run_cmd(["docker", "compose", "up", "--build", "-d"])
    if success:
        print("\n✅ Smart Helmet Live Server deployed successfully in Docker!")
        print("🌐 Open dashboard: http://localhost:5050")
        print("📊 Telemetry API: http://localhost:5050/api/status")
    return success

def run_tests():
    print("\n🧪 Running Smart Helmet System Test Suite...")
    cmd = [sys.executable, "tests/test_system.py", "--test", "all"]
    return run_cmd(cmd)

def main():
    parser = argparse.ArgumentParser(description="Smart Helmet Deployment Manager")
    parser.add_argument("--mode", choices=["local", "docker", "test"], default="local",
                        help="Deployment mode: local, docker, or test")
    parser.add_argument("--port", type=int, default=5050, help="Port for local server")
    args = parser.parse_args()

    if args.mode == "local":
        deploy_local(args.port)
    elif args.mode == "docker":
        deploy_docker()
    elif args.mode == "test":
        run_tests()

if __name__ == "__main__":
    main()
