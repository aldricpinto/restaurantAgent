import os
import sys
import uuid
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from utils.logger import server_logger as logger

load_dotenv()

mcp = FastMCP("Ophelia")

OPHELIA_BASE = os.getenv("OPHELIA_BASE_URL")
OPHELIA_KEY = os.getenv("OPHELIA_API_KEY")



def headers(idempotency_key: str = None):
    h = {
        "Authorization": f"Bearer {OPHELIA_KEY}",
        "Content-Type": "application/json"
    }
    if idempotency_key:
        h["Idempotency-Key"] = idempotency_key
    return h


@mcp.tool()
def search_venues(
    vertical: str,
    term: str,
    location: str,
    datetime: str,
    party_size: int = 2
) -> dict:
    """
    Search for venues on Ophelia.

    vertical: 'dining', 'fitness', or 'entertainment'
    term: what you're looking for e.g. 'sushi', 'yoga', 'Hamilton'
    location: city and state e.g. 'New York, NY'
    datetime: ISO format e.g. '2026-06-21T20:00:00'
    party_size: number of people (required for dining and entertainment)
    """
    logger.info(f"search_venues: vertical={vertical!r}, term={term!r}, location={location!r}, datetime={datetime!r}, party_size={party_size}")
    with httpx.Client(timeout=90) as client:
        response = client.post(
            f"{OPHELIA_BASE}/venues/search",
            headers=headers(),
            json={
                "vertical": vertical,
                "term": term,
                "location": location,
                "datetime": datetime,
                "party_size": party_size
            }
        )
        logger.info(f"search_venues: status_code={response.status_code}")
        try:
            res_json = response.json()
            logger.info(f"search_venues: returned {len(res_json.get('venues', []))} venues")
            logger.debug(f"search_venues full response: {res_json}")
            return res_json
        except Exception as e:
            logger.error(f"search_venues parsing failed: {e}. Raw content: {response.text}")
            raise


@mcp.tool()
def create_booking(
    venue_id: str,
    datetime: str,
    party_size: int,
    customer_name: str,
    customer_email: str,
    customer_phone: str,
    vertical: str = "dining",
    password: str = None,
    card_number: str = None,
    card_exp_month: str = None,
    card_exp_year: str = None,
    card_cvv: str = None,
    card_name: str = None,
    card_postal: str = None
) -> dict:
    """
    Create a booking at a venue.

    Returns status 'requires_action' when OTP confirmation is needed.
    Pass card details for dining (optional hold) and fitness (required).
    """
    logger.info(
        f"create_booking: venue_id={venue_id!r}, datetime={datetime!r}, party_size={party_size}, "
        f"customer={customer_name!r}/{customer_email!r}/{customer_phone!r}, vertical={vertical!r}"
    )
    party_size = int(party_size)
    payload = {
        "vertical": vertical,
        "venue_id": venue_id,
        "datetime": datetime,
        "customer": {
            "name": customer_name,
            "email": customer_email,
            "phone_number": customer_phone
        },
        "metadata":{}
    }

    if vertical == 'dining' or vertical == 'entertainment':
        payload["party_size"] = party_size
    

    if card_number:
        logger.info("create_booking: Payment card provided")
        payload["metadata"] = {
            "payment": {
                "card_number": card_number,
                "exp_month": card_exp_month,
                "exp_year": card_exp_year,
                "cvv": card_cvv,
                "name_on_card": card_name,
                "postal_code": card_postal
            }
        }

    if vertical == 'fitness':
        payload['metadata']['password'] = password

    with httpx.Client(timeout=200) as client:
        idempotency_key = str(uuid.uuid4())
        logger.info(f"create_booking: posting request with idempotency_key={idempotency_key}")
        response = client.post(
            f"{OPHELIA_BASE}/bookings",
            headers=headers(idempotency_key=idempotency_key),
            json=payload
        )
        logger.info(f"create_booking: status_code={response.status_code}")
        try:
            res_json = response.json()
            logger.info(f"create_booking response details: id={res_json.get('id')}, status={res_json.get('status')}")
            logger.debug(f"create_booking response body: {res_json}")
            return res_json
        except Exception as e:
            logger.error(f"create_booking parsing failed: {e}. Raw content: {response.text}")
            raise


@mcp.tool()
def continue_booking(booking_id: str, otp_code: str) -> dict:
    """
    Submit OTP code to confirm a booking that has status 'requires_action'.

    booking_id: the bkg_ id from create_booking
    otp_code: the code the customer received via SMS or email
    """
    logger.info(f"continue_booking: booking_id={booking_id!r}, otp_code={otp_code!r}")
    with httpx.Client(timeout=30) as client:
        response = client.post(
            f"{OPHELIA_BASE}/bookings/{booking_id}/continue",
            headers=headers(),
            json={"otp_code": otp_code}
        )
        logger.info(f"continue_booking: status_code={response.status_code}")
        try:
            res_json = response.json()
            logger.info(f"continue_booking response details: id={res_json.get('id')}, status={res_json.get('status')}")
            logger.debug(f"continue_booking response body: {res_json}")
            return res_json
        except Exception as e:
            logger.error(f"continue_booking parsing failed: {e}. Raw content: {response.text}")
            raise


@mcp.tool()
def get_booking(booking_id: str) -> dict:
    """
    Get the current status and details of a booking.

    booking_id: the bkg_ id from create_booking
    """
    logger.info(f"get_booking: booking_id={booking_id!r}")
    with httpx.Client(timeout=30) as client:
        response = client.get(
            f"{OPHELIA_BASE}/bookings/{booking_id}",
            headers=headers()
        )
        logger.info(f"get_booking: status_code={response.status_code}")
        try:
            res_json = response.json()
            logger.info(f"get_booking response details: id={res_json.get('id')}, status={res_json.get('status')}")
            logger.debug(f"get_booking response body: {res_json}")
            return res_json
        except Exception as e:
            logger.error(f"get_booking parsing failed: {e}. Raw content: {response.text}")
            raise


@mcp.tool()
def cancel_booking(booking_id: str) -> dict:
    """
    Cancel an existing booking.

    booking_id: the bkg_ id from create_booking
    """
    logger.info(f"cancel_booking: booking_id={booking_id!r}")
    with httpx.Client(timeout=30) as client:
        response = client.post(
            f"{OPHELIA_BASE}/bookings/{booking_id}/cancel",
            headers=headers()
        )
        logger.info(f"cancel_booking: status_code={response.status_code}")
        try:
            res_json = response.json()
            logger.info(f"cancel_booking response details: id={res_json.get('id')}, status={res_json.get('status')}")
            logger.debug(f"cancel_booking response body: {res_json}")
            return res_json
        except Exception as e:
            logger.error(f"cancel_booking parsing failed: {e}. Raw content: {response.text}")
            raise


if __name__ == "__main__":
    mcp.run()