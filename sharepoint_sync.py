import os
import requests
from msal import ConfidentialClientApplication
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# === Load secrets from environment ===
load_dotenv()
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")

# === SharePoint Info ===
DRIVE_ID = "b!rsvKwTOMNUeCRjsRDMZh-kprgBi3tc1LiWVKbiOrmtWWapTcFH-5QLtKqb12SEmT"
FOLDER_PATH = os.getenv("SP_FOLDER_PATH", "AI")

# === Persistent storage ===
BASE_STORAGE = "/data"
SYNC_TIER = os.getenv("SYNC_TIER", "public").lower()

if SYNC_TIER not in ("public", "internal"):
    raise ValueError("SYNC_TIER must be 'public' or 'internal'")

DESTINATION_FOLDER = os.path.join(BASE_STORAGE, "documents", SYNC_TIER)
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


def list_children(headers, folder_path: str):
    encoded_path = folder_path.replace(" ", "%20")
    url = f"https://graph.microsoft.com/v1.0/drives/{DRIVE_ID}/root:/{encoded_path}:/children"

    all_items = []
    while url:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        all_items.extend(data.get("value", []))
        url = data.get("@odata.nextLink")  # pagination
    return all_items



def sync_sharepoint():
    access_token = authenticate()
    headers = {"Authorization": f"Bearer {access_token}"}

    def walk(folder_path: str):
        items = list_children(headers, folder_path)
        for item in items:
            name = item.get("name", "")
            item_path = f"{folder_path}/{name}"

            # If folder → recurse
            if item.get("folder"):
                walk(item_path)
                continue

            # If file → download
            if name.lower().endswith((".pdf", ".docx")):
                print(f"Downloading: {item_path}")
                download_url = item.get("@microsoft.graph.downloadUrl")
                if not download_url:
                    continue

                file_data = requests.get(download_url)
                file_data.raise_for_status()

                # Mirror folder structure locally
                rel_dir = folder_path.replace(FOLDER_PATH, "").lstrip("/")
                local_dir = Path(DESTINATION_FOLDER) / rel_dir
                local_dir.mkdir(parents=True, exist_ok=True)

                dest_path = local_dir / name
                with open(dest_path, "wb") as f:
                    f.write(file_data.content)

    walk(FOLDER_PATH)
    print(f"✅ Sync complete. Files saved to {DESTINATION_FOLDER}")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/sync-now")
def trigger_sync():
    try:
        sync_sharepoint()
        return {"status": "success", "message": "SharePoint sync completed."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    sync_sharepoint()
