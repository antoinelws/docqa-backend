import os
import requests
from msal import ConfidentialClientApplication
from pathlib import Path
from dotenv import load_dotenv

# === Load secrets from environment ===
load_dotenv()
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")

# === SharePoint Info ===
SHAREPOINT_SITE = "erpintegratedsolutions.sharepoint.com"
SHAREPOINT_SITE_NAME = "PMO"
DOCUMENT_LIBRARY = "Documents"
FOLDER_PATH = "AI"

# === Backend processing path ===
DESTINATION_FOLDER = "documents"
os.makedirs(DESTINATION_FOLDER, exist_ok=True)

# === Authentication ===
SCOPES = ["https://graph.microsoft.com/.default"]
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"

def authenticate():
    app = ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET,
    )
    result = app.acquire_token_for_client(scopes=SCOPES)
    if "access_token" not in result:
        raise Exception("Authentication failed.")
    return result["access_token"]

def sync_sharepoint():
    access_token = authenticate()
    headers = {"Authorization": f"Bearer {access_token}"}

    # === Resolve Site ID ===
    site_resp = requests.get(
        f"https://graph.microsoft.com/v1.0/sites/{SHAREPOINT_SITE}:/sites/{SHAREPOINT_SITE_NAME}",
        headers=headers
    )
    site_resp.raise_for_status()
    site_id = site_resp.json()["id"]

    # === Get Folder Items ===
    drive_resp = requests.get(
        f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{DOCUMENT_LIBRARY}/{FOLDER_PATH}:/children",
        headers=headers
    )
    drive_resp.raise_for_status()
    files = drive_resp.json().get("value", [])

    # === Download Supported Files ===
    for file in files:
        name = file.get("name")
        if name.endswith(".pdf") or name.endswith(".docx"):
            print(f"Downloading: {name}")
            download_url = file.get("@microsoft.graph.downloadUrl")
            file_data = requests.get(download_url)
            dest_path = Path(DESTINATION_FOLDER) / name
            with open(dest_path, "wb") as f:
                f.write(file_data.content)

    print("✅ Sync complete. Files saved to ./documents/")

# Auto-run on backend startup
try:
    print("🚀 Running SharePoint sync...")
    sync_sharepoint()
except Exception as e:
    print(f"❌ SharePoint sync failed: {e}")
