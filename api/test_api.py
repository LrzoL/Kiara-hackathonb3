#!/usr/bin/env python3
"""
Test script for GitHub Documentation Agent API.
Demonstrates how to use the FastAPI endpoints.
"""

import asyncio
import json
import time
import httpx

API_BASE_URL = "http://localhost:8000"

async def test_api():
    """Test the API endpoints."""
    
    print("🧪 Testing GitHub Documentation Agent API")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        
        # Test 1: Root endpoint
        print("\n[1/6] Testing root endpoint...")
        try:
            response = await client.get(f"{API_BASE_URL}/")
            print(f"✅ Status: {response.status_code}")
            print(f"📄 Response: {response.json()}")
        except Exception as e:
            print(f"❌ Error: {e}")
            return
        
        # Test 2: Health check
        print("\n[2/6] Testing health endpoint...")
        try:
            response = await client.get(f"{API_BASE_URL}/health")
            print(f"✅ Status: {response.status_code}")
            health_data = response.json()
            print(f"🏥 Health: {health_data['status']}")
            print(f"🔧 Components: {health_data['components']}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 3: Status endpoint
        print("\n[3/6] Testing status endpoint...")
        try:
            response = await client.get(f"{API_BASE_URL}/status")
            print(f"✅ Status: {response.status_code}")
            status_data = response.json()
            print(f"🚀 API Version: {status_data.get('api_version')}")
            print(f"🔧 Features: {len(status_data.get('features', []))}")
            print(f"🌐 Languages: {len(status_data.get('supported_languages', []))}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 4: Documentation generation (POST method)
        print("\n[4/6] Testing documentation generation (POST)...")
        try:
            test_repo_url = "https://github.com/athospugliese/testingrobertaai"
            
            request_data = {
                "repository_url": test_repo_url,
                "include_vision": True,
                "output_format": "markdown"
            }
            
            print(f"🔄 Generating docs for: {test_repo_url}")
            start_time = time.time()
            
            response = await client.post(
                f"{API_BASE_URL}/generate-docs",
                json=request_data
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Status: {response.status_code}")
            print(f"⏱️  Request time: {elapsed:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                print(f"📈 Success: {result['success']}")
                print(f"⚡ Generation time: {result['generation_time']:.2f}s")
                
                if result['success']:
                    analysis = result.get('analysis', {})
                    print(f"🔍 Language: {analysis.get('language')}")
                    print(f"📦 Frameworks: {', '.join(analysis.get('frameworks', []))}")
                    print(f"🏗️  Project type: {analysis.get('project_type')}")
                    print(f"📄 README length: {len(result.get('documentation', ''))} chars")
                    
                    # Save generated README
                    readme_content = result.get('documentation', '')
                    if readme_content:
                        with open('generated_README_via_API.md', 'w', encoding='utf-8') as f:
                            f.write(readme_content)
                        print(f"💾 README saved to: generated_README_via_API.md")
                else:
                    print(f"❌ Errors: {result.get('errors', [])}")
                    print(f"⚠️  Warnings: {result.get('warnings', [])}")
            else:
                print(f"❌ Error response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 5: Documentation generation (GET method)
        print("\n[5/6] Testing documentation generation (GET)...")
        try:
            response = await client.get(f"{API_BASE_URL}/generate-docs/athospugliese/testingrobertaai")
            print(f"✅ Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"📈 Success: {result['success']}")
                print(f"⚡ Generation time: {result['generation_time']:.2f}s")
            else:
                print(f"❌ Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 6: Invalid repository URL
        print("\n[6/6] Testing invalid repository URL...")
        try:
            invalid_request = {
                "repository_url": "https://invalid-site.com/user/repo",
                "include_vision": False
            }
            
            response = await client.post(
                f"{API_BASE_URL}/generate-docs",
                json=invalid_request
            )
            
            print(f"📊 Status: {response.status_code} (expected 422)")
            if response.status_code == 422:
                print("✅ Validation working correctly")
            else:
                print(f"📄 Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 API testing completed!")

async def main():
    """Main test function."""
    print("Starting API server test...")
    print("Make sure the API server is running with: python api/main.py")
    print("Or: uvicorn api.main:app --reload")
    
    await test_api()

if __name__ == "__main__":
    asyncio.run(main())