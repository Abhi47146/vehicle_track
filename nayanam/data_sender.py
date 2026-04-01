import redis
import json
import httpx
from nayanam.logger import Logger

logger = Logger('data_send')

headers = {"Content-Type": "application/json"}
client = httpx.Client(
    timeout=httpx.Timeout(1.0),  # short timeout
    headers=headers
)

def send_to_redis(vbv_file, channel="detector"):
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    logger.info(f"Generated vbv_file {vbv_file}")
    redis_client.publish("detector", json.dumps(vbv_file))

def send_to_api(vbv_file, api_url):
    try:
        response = client.post(api_url, json=vbv_file, headers=headers)
        response.raise_for_status()
        logger.info("Data sent successfully to API")

    except httpx.TimeoutException:
        logger.error("API request timed out")

    except httpx.HTTPStatusError as e:
        logger.error(f"API error {e.response.status_code}: {e.response.text}")

    except Exception:
        logger.exception("Unexpected error while sending vehicle data")
