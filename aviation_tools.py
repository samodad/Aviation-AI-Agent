"""
Aviation Tools for SkyMind SL
Tools used by the LangGraph agents to fetch aviation data.
"""

import os
import requests
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Sri Lanka Airport ICAO / IATA Reference Data
# ──────────────────────────────────────────────
SL_AIRPORTS = {
    "VCBI": {
        "name": "Bandaranaike International Airport",
        "iata": "CMB",
        "city": "Colombo (Katunayake)",
        "elevation_ft": 30,
        "lat": 7.1807,
        "lon": 79.8842,
        "runways": ["04/22 (3350m)", "13/31 (2000m)"],
        "info": "Main international gateway to Sri Lanka. Operated by Airport & Aviation Services (Sri Lanka) Ltd.",
    },
    "VCCA": {
        "name": "Mattala Rajapaksa International Airport",
        "iata": "HRI",
        "city": "Hambantota (Mattala)",
        "elevation_ft": 157,
        "lat": 6.2847,
        "lon": 81.1241,
        "runways": ["04/22 (3500m)"],
        "info": "Sri Lanka's second international airport. Located in the deep south.",
    },
    "VCCB": {
        "name": "Ratmalana Airport",
        "iata": "RML",
        "city": "Colombo (Ratmalana)",
        "elevation_ft": 22,
        "lat": 6.8220,
        "lon": 79.8862,
        "runways": ["04/22 (2008m)"],
        "info": "Domestic airport and Sri Lanka Air Force base. General aviation hub.",
    },
    "VCCT": {
        "name": "China Bay Airport",
        "iata": "TRR",
        "city": "Trincomalee",
        "elevation_ft": 6,
        "lat": 8.5385,
        "lon": 81.1819,
        "runways": ["03/21 (1740m)"],
        "info": "Domestic airport in the Eastern Province.",
    },
    "VCCJ": {
        "name": "Jaffna International Airport",
        "iata": "JAF",
        "city": "Jaffna (Palaly)",
        "elevation_ft": 34,
        "lat": 9.7924,
        "lon": 80.0701,
        "runways": ["05/23 (1800m)"],
        "info": "Domestic airport serving the Northern Province.",
    },
}

# ──────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────

@tool
def get_airport_info(icao_or_iata: str) -> str:
    """
    Get detailed information about a Sri Lankan airport.
    Accepts ICAO code (e.g., VCBI) or IATA code (e.g., CMB).
    Returns airport name, location, runways, elevation, and operational info.
    """
    code = icao_or_iata.upper().strip()

    # Try direct ICAO lookup
    if code in SL_AIRPORTS:
        ap = SL_AIRPORTS[code]
        return (
            f"✈️ **{ap['name']}** ({code} / {ap['iata']})\n"
            f"- Location: {ap['city']}, Sri Lanka\n"
            f"- Elevation: {ap['elevation_ft']} ft AMSL\n"
            f"- Coordinates: {ap['lat']}°N, {ap['lon']}°E\n"
            f"- Runways: {', '.join(ap['runways'])}\n"
            f"- Info: {ap['info']}"
        )

    # Try IATA reverse lookup
    for icao, ap in SL_AIRPORTS.items():
        if ap["iata"] == code:
            return (
                f"✈️ **{ap['name']}** ({icao} / {ap['iata']})\n"
                f"- Location: {ap['city']}, Sri Lanka\n"
                f"- Elevation: {ap['elevation_ft']} ft AMSL\n"
                f"- Coordinates: {ap['lat']}°N, {ap['lon']}°E\n"
                f"- Runways: {', '.join(ap['runways'])}\n"
                f"- Info: {ap['info']}"
            )

        known = ", ".join([f"{v['iata']}/{k}" for k, v in SL_AIRPORTS.items()])
    return f"Airport '{icao_or_iata}' not found. Known airports: {known}"


@tool
def get_weather_briefing(location: str) -> str:
    """
    Get current weather conditions for aviation briefing at a Sri Lankan location or airport.
    Returns temperature, wind, visibility, humidity, and cloud cover relevant for pilots.
    Accepts city names or airport IATA/ICAO codes.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    # Map ICAO/IATA to city for weather lookup
    location_map = {
        "VCBI": "Katunayake", "CMB": "Katunayake",
        "VCCA": "Hambantota", "HRI": "Hambantota",
        "VCCB": "Ratmalana", "RML": "Colombo",
        "VCCT": "Trincomalee", "TRR": "Trincomalee",
        "VCCJ": "Jaffna", "JAF": "Jaffna",
    }

    city = location_map.get(location.upper(), location)

    if not api_key or api_key == "your_openweather_api_key_here":
        # Return simulated briefing for demo purposes
        return _simulated_weather_briefing(location, city)

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {"q": f"{city},LK", "appid": api_key, "units": "metric"}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if resp.status_code != 200:
            return _simulated_weather_briefing(location, city)

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"].get("speed", 0) * 1.944  # m/s to knots
        wind_deg = data["wind"].get("deg", 0)
        visibility_m = data.get("visibility", 9999)
        visibility_km = visibility_m / 1000
        weather_desc = data["weather"][0]["description"].title()
        pressure = data["main"]["pressure"]

        wind_dir = _degrees_to_compass(wind_deg)

        vfr_status = "✅ VFR Conditions" if visibility_km >= 5 else "⚠️ Marginal VFR / IFR Conditions"

        return (
            f"🌤️ **Aviation Weather Briefing — {city}, Sri Lanka**\n"
            f"- Conditions: {weather_desc}\n"
            f"- Temperature: {temp:.1f}°C\n"
            f"- Wind: {wind_dir} at {wind_speed:.0f} kts ({wind_deg}°)\n"
            f"- Visibility: {visibility_km:.1f} km\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- VFR Status: {vfr_status}\n"
            f"*Always consult official METAR/TAF from AASL before flight.*"
        )
    except Exception as e:
        return _simulated_weather_briefing(location, city)


def _simulated_weather_briefing(location: str, city: str) -> str:
    """Returns a realistic simulated weather briefing for demo purposes."""
    return (
        f"🌤️ **Aviation Weather Briefing — {city}, Sri Lanka** *(Simulated Demo)*\n"
        f"- Conditions: Partly Cloudy with Tropical Haze\n"
        f"- Temperature: 28°C (Dewpoint: 24°C)\n"
        f"- Wind: SW at 12 kts (220°)\n"
        f"- Visibility: 8 km (reducing to 4 km in rain showers)\n"
        f"- Humidity: 82%\n"
        f"- Pressure: 1011 hPa (QNH)\n"
        f"- Cloud: FEW018 SCT025 BKN060\n"
        f"- VFR Status: ✅ VFR Conditions (monitor for afternoon CB development)\n"
        f"*This is a demo response. Add your OPENWEATHER_API_KEY for live data.*\n"
        f"*Always consult official METAR/TAF from AASL / Meteorology Dept before flight.*"
    )


@tool
def get_notam_info(topic: str) -> str:
    """
    Get NOTAM (Notice to Air Missions) information and guidance relevant to Sri Lankan airspace.
    Useful for questions about temporary restrictions, airspace changes, or NOTAM procedures.
    """
    notam_knowledge = {
        "fir": (
            "📋 **Sri Lanka FIR (Flight Information Region)**\n"
            "- FIR Callsign: COLOMBO CONTROL\n"
            "- ICAO FIR ID: VCCF\n"
            "- Colombo ACC frequency: 128.9 MHz (primary)\n"
            "- AIP Sri Lanka is maintained by AASL (Airport & Aviation Services)\n"
            "- NOTAMs are issued via the international NOTAM system (ICAO Annex 15)"
        ),
        "restricted": (
            "📋 **Restricted Airspace in Sri Lanka**\n"
            "- VCR001–VCR010: Various restricted areas over military installations\n"
            "- Presidential Palace / Parliament area: Permanent no-fly zone\n"
            "- Sri Lanka Air Force bases (Katunayake, Ratmalana, China Bay): Coordination required\n"
            "- Contact Colombo ACC for current status of restricted areas."
        ),
        "drone": (
            "📋 **UAS / Drone Regulations in Sri Lanka**\n"
            "- Regulated by Civil Aviation Authority of Sri Lanka (CAASL)\n"
            "- Drones above 250g require registration\n"
            "- BVLOS operations require special approval\n"
            "- Flying within 3 nm of any airport is prohibited without CAASL clearance\n"
            "- Contact: caasl.gov.lk for permits"
        ),
    }

    topic_lower = topic.lower()
    for key, value in notam_knowledge.items():
        if key in topic_lower:
            return value

    return (
        f"📋 **NOTAM Guidance — '{topic}'**\n"
        "For current NOTAMs affecting Sri Lankan airspace:\n"
        "1. Visit the AASL AIS Portal: https://www.airport.lk\n"
        "2. Use ICAO NAIP (Notice to Airmen Information Publication)\n"
        "3. Contact Colombo Approach (128.9 MHz) for real-time updates\n"
        "4. Check CAASL website: https://www.caasl.gov.lk\n\n"
        "The Colombo FIR (VCCF) issues NOTAMs for:\n"
        "- Airspace restrictions (temporary / permanent)\n"
        "- Runway/taxiway closures at VCBI, VCCA, VCCB\n"
        "- Navigation aid outages (VOR, ILS, NDB)\n"
        "- Military exercises affecting civil aviation"
    )


@tool
def get_aviation_regulations(topic: str) -> str:
    """
    Get information about aviation regulations, procedures, and requirements in Sri Lanka.
    Topics include: PPL, CPL, ATPL licensing, medical requirements, FPL filing, ATC procedures.
    """
    regulations = {
        "ppl": (
            "📚 **Private Pilot License (PPL) — Sri Lanka**\n"
            "- Issued by: Civil Aviation Authority of Sri Lanka (CAASL)\n"
            "- Minimum age: 17 years\n"
            "- Medical: Class 2 Medical Certificate\n"
            "- Flight hours: Minimum 45 hours (inc. 10 hrs solo)\n"
            "- Exams: Air Law, Meteorology, Navigation, Aircraft General Knowledge\n"
            "- Training: Approved Flying Training Organizations (FTO) in Sri Lanka\n"
            "- Reference: CAASL ANR Part 61"
        ),
        "cpl": (
            "📚 **Commercial Pilot License (CPL) — Sri Lanka**\n"
            "- Issued by: CAASL\n"
            "- Minimum age: 18 years\n"
            "- Medical: Class 1 Medical Certificate\n"
            "- Flight hours: Minimum 200 hours total time\n"
            "- Exams: Advanced ATPL subjects (14 papers)\n"
            "- Reference: CAASL ANR Part 61"
        ),
        "fpl": (
            "📚 **Flight Plan Filing — Sri Lanka**\n"
            "- File via: AFTN or Colombo ARO (Aerodrome Reporting Office)\n"
            "- Format: ICAO Doc 4444 (PANS-ATM)\n"
            "- IFR: File at least 60 minutes before ETD\n"
            "- VFR: File at least 30 minutes before ETD\n"
            "- ARO Contact: +94 11 225 2861 (VCBI)\n"
            "- Online filing: SITA AeroBahn or Homebriefing available for registered operators"
        ),
        "medical": (
            "📚 **Aviation Medical Requirements — Sri Lanka**\n"
            "- Class 1: Required for CPL/ATPL holders (airline pilots)\n"
            "- Class 2: Required for PPL holders\n"
            "- Issued by: CAASL-approved Aviation Medical Examiners (AME)\n"
            "- Validity: Class 1 — 12 months (6 months if over 40); Class 2 — 24 months\n"
            "- Reference: CAASL ANR Part 67"
        ),
    }

    topic_lower = topic.lower()
    for key, value in regulations.items():
        if key in topic_lower:
            return value

    return (
        f"📚 **Aviation Regulations — '{topic}'**\n"
        "Sri Lanka aviation is governed by:\n"
        "- **CAASL** (Civil Aviation Authority of Sri Lanka) — caasl.gov.lk\n"
        "- **AASL** (Airport & Aviation Services Ltd) — airport.lk\n"
        "- **ANR** (Air Navigation Regulations) based on ICAO Standards\n\n"
        "Key regulation areas:\n"
        "- Pilot licensing (PPL, CPL, ATPL, IR, ME)\n"
        "- Aircraft registration and airworthiness\n"
        "- ATC and airspace management\n"
        "- Drone / UAS operations\n"
        "- Airline operations and AOC\n\n"
        "For specific regulations, visit: https://www.caasl.gov.lk/index.php/regulations"
    )


@tool
def search_aviation_web(query: str) -> str:
    """
    Search the web for current aviation news, flight schedules, and real-time information
    related to Sri Lankan aviation. Use for questions about specific airlines, current events,
    flight delays, or anything requiring up-to-date information.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key or tavily_key == "your_tavily_api_key_here":
        return (
            f"🔍 Web search for: '{query}'\n"
            "*(Web search requires a TAVILY_API_KEY — get one free at tavily.com)*\n\n"
            "Based on general knowledge:\n"
            "- SriLankan Airlines (UL) is the national carrier, hub at VCBI (CMB)\n"
            "- FitsAir operates domestic routes from Ratmalana (VCCB)\n"
            "- VCBI handles 10+ million passengers annually\n"
            "- Key airlines at CMB: Emirates, Qatar Airways, Singapore Airlines, Air India, IndiGo"
        )

    try:
        search = TavilySearchResults(max_results=3, tavily_api_key=tavily_key)
        results = search.invoke(f"{query} Sri Lanka aviation")
        if results:
            output = f"🔍 **Web Search Results for: '{query}'**\n\n"
            for i, r in enumerate(results[:3], 1):
                output += f"{i}. [{r.get('title', 'Result')}]({r.get('url', '')})\n"
                content = r.get('content', '')[:300]
                output += f"   {content}...\n\n"
            return output
        return f"No results found for '{query}'."
    except Exception as e:
        return f"Search error: {str(e)}. Please check your TAVILY_API_KEY."


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def _degrees_to_compass(degrees: float) -> str:
    """Convert wind degrees to compass direction."""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(degrees / 22.5) % 16
    return directions[idx]


# Export all tools as a list
ALL_TOOLS = [
    get_airport_info,
    get_weather_briefing,
    get_notam_info,
    get_aviation_regulations,
    search_aviation_web,
]