import os

from sentinelhub import SHConfig

# CDSE deployment base URL (Copernicus Data Space)
CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"

# CDSE identity (token) endpoint (official CDSE realm)
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)


def get_config():
    """
    Retrieves Sentinel Hub configuration using environment variables.
    Requires SH_CLIENT_ID and SH_CLIENT_SECRET to be set.
    """
    client_id = os.environ.get("SH_CLIENT_ID")
    client_secret = os.environ.get("SH_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Sentinel Hub credentials not found. "
            "Please set SH_CLIENT_ID and SH_CLIENT_SECRET environment variables."
        )

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url = CDSE_BASE_URL
    config.sh_token_url = CDSE_TOKEN_URL
    config.sh_process_api_url = f"{CDSE_BASE_URL}/api/v1/process"

    return config
