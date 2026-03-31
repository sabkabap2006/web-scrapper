from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import logging
from scraper_core import scrape_website, invoke_ai_agent, CRISIS_KEYWORDS

app = FastAPI(
    title="Web Scrapping AI Agent API",
    description="Backend API for scraping websites and extracting structured information using Nvidia API.",
    version="1.0.0"
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ROOT — quick health check when you open localhost:8000
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return """
    <html><body style="font-family:sans-serif;padding:40px;background:#0f0f1a;color:#e2e8f0">
    <h1 style="color:#7c3aed">🕵️ Web Scrapping AI Agent API</h1>
    <p>Server is running. Available browser-accessible GET endpoints:</p>
    <ul>
      <li><a style="color:#a78bfa" href="/api/scrape/news">📰 /api/scrape/news</a> — WION News + India Today (default)</li>
      <li><a style="color:#a78bfa" href="/api/scrape/supply-chain?search_query=global supply chain disruptions 2026">/api/scrape/supply-chain?search_query=...</a></li>
      <li><a style="color:#a78bfa" href="/api/scrape/world-bank">/api/scrape/world-bank</a></li>
      <li><a style="color:#a78bfa" href="/api/scrape/smart-resource">/api/scrape/smart-resource</a></li>
      <li><a style="color:#a78bfa" href="/api/scrape/custom">/api/scrape/custom</a> — WION News + India Today (default)</li>
    </ul>
    <p>Interactive docs: <a style="color:#a78bfa" href="/docs">/docs</a></p>
    </body></html>
    """

# Request Models
class SmartResourceRequest(BaseModel):
    url: str
    allocation_objective: str
    use_browser: Optional[bool] = False

class SupplyChainRequest(BaseModel):
    search_query: str
    use_browser: Optional[bool] = False

class WorldBankRequest(BaseModel):
    use_browser: Optional[bool] = False

class CustomScrapeRequest(BaseModel):
    urls: list[str]          # list of URLs to scrape (replaces single url field)
    prompt: str
    use_browser: Optional[bool] = False

# Endpoints
@app.post("/api/scrape/smart-resource", summary="Smart Resource Allocation AI")
def scrape_smart_resource(req: SmartResourceRequest):
    user_prompt = f"""You are a Smart Resource Allocation AI.
First, perform a highly detailed extraction of resources from the target webpage. Pay special attention to specific physical, economic, or agricultural resources like fuel, water, petroleum, wheat, rice, etc., and explicitly detail their availability, quantity, and characteristics.
Then, analyze these extracted resources to fulfill the following objective: "{req.allocation_objective}"

Please present your output as a highly descriptive, well-structured Markdown report. Do NOT use JSON format. 
Your report should include:
- **Executive Summary:** A high-level overview of the findings and your proposed strategy.
- **Extracted Resources:** A very detailed, descriptive breakdown of each physical/economic resource found (e.g., water, petroleum, wheat, tech), clearly stating specific availability, historical levels, quantity or metrics, and whether it's scarce or abundant.
- **Allocation Strategy:** Your strategic plan detailing target recipients, allocated values/percentages, and comprehensive justifications for each allocation based on the available resources data."""

    target_url = req.url
    if not target_url.startswith("http://") and not target_url.startswith("https://"):
        target_url = "https://" + target_url

    errors = []
    def error_cb(err):
        errors.append(err)

    text = scrape_website(target_url, use_browser=req.use_browser, error_callback=error_cb)
    if errors and not text:
        raise HTTPException(status_code=500, detail=" | ".join(errors))
    
    if not text:
        raise HTTPException(status_code=404, detail="No text could be extracted from the target.")
    
    scraped_text = f"\\n\\n--- Source: {target_url} ---\\n{text}"
    
    try:
        # We don't stream for the REST API
        ai_response = invoke_ai_agent(scraped_text, user_prompt)
        return {"report": ai_response}
    except Exception as e:
        logger.error(f"AI API Error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred with the AI API: {str(e)}")

@app.post("/api/scrape/supply-chain", summary="Supply Chain Disruption")
def scrape_supply_chain(req: SupplyChainRequest):
    user_prompt = """Extract supply chain disruption information from the provided text contexts. 
CRITICAL RULE: You must STRICTLY limit your extraction to disruptions that actally occur in or are ongoing in the year 2026. Ignore any purely historical events from prior years (2025 and earlier).

You MUST return your answer strictly as a JSON array of objects, where each object strictly matches this schema:
{
  "source": "string (name of the website/news outlet providing the insight)",
  "timestamp_reported": "string (date and time)",
  "timestamp_scraped": "string (ISO timestamp)",
  "trend_score": "number",
  "disruption_id": "number",
  "disruption_type": "string (e.g., geopolitical, weather, strike)",
  "industry": "string",
  "supplier_tier": "string (e.g., tier_1)",
  "supplier_region": "string (e.g., global, middle_east, panama)",
  "supplier_size": "string",
  "has_backup_supplier": "boolean",
  "disruption_severity": "string (low, medium, high)",
  "production_impact_pct": "number",
  "revenue_loss_usd": "number",
  "response_type": "string (mitigation, avoidance, acceptance)",
  "response_time_days": "number",
  "partial_recovery_days": "number",
  "full_recovery_days": "number",
  "permanent_supplier_change": "boolean",
  "week": "number",
  "port": "string (name of port or 'unknown')",
  "region": "string",
  "shipping_mode": "string (e.g., sea, air, rail)",
  "avg_wait_days": "number",
  "disruption_index": "number",
  "freight_rate_usd": "number",
  "fuel_price_usd": "number",
  "backlog_teu": "number",
  "on_time_pct": "number",
  "heat_score": "number"
}"""

    scraped_text = ""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    errors = []
    def error_cb(err):
        errors.append(err)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(req.search_query, max_results=3))
            
            if not results:
                raise HTTPException(status_code=404, detail="DuckDuckGo returned 0 results! Query might be invalid or IP rate-limited.")
                
            for idx, res in enumerate(results):
                link = res.get("href")
                title = res.get("title", "Unknown Title")
                snippet = res.get("body", "No search snippet available.")
                
                if link:
                    text = scrape_website(link, use_browser=req.use_browser, error_callback=error_cb)
                    scraped_text += f"\\n\\n--- Source {idx+1}: {title} ({link}) ---\\n"
                    if text and len(text) > 150:
                        scraped_text += f"Extracted Text:\\n{text[:12000]}\\n"
                    else:
                        scraped_text += f"Search Snippet:\\n{snippet}\\n"
                        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Web Search failed: {e}")

    try:
        ai_response = invoke_ai_agent(scraped_text, user_prompt)
        
        # Try to parse the JSON array. Since LLM might wrap in ```json, we do a quick cleanup
        clean_res = ai_response.strip()
        if clean_res.startswith("```json"):
            clean_res = clean_res[7:]
        if clean_res.startswith("```"):
            clean_res = clean_res[3:]
        if clean_res.endswith("```"):
            clean_res = clean_res[:-3]
            
        json_data = json.loads(clean_res.strip())
        return {"disruptions": json_data}
    except json.JSONDecodeError:
        # If it failed to decode, just return the raw text
        return {"raw_output": ai_response, "warning": "Failed to parse JSON array from AI output."}
    except Exception as e:
        logger.error(f"AI API Error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred with the AI API: {str(e)}")

@app.post("/api/scrape/world-bank", summary="World Bank Automated")
def scrape_world_bank(req: WorldBankRequest):
    url = "https://data.worldbank.org/"
    user_prompt = '''Extract all available resources, datasets, reports, and indicators from the page. 
You MUST return your answer strictly as a JSON object, grouped region-wise. 
Example JSON structure:
{
  "Global": [ {"title": "...", "description": "...", "link": "..."} ],
  "South Asia": [ {"title": "...", "description": "...", "link": "..."} ]
}'''

    errors = []
    def error_cb(err):
        errors.append(err)

    text = scrape_website(url, use_browser=req.use_browser, error_callback=error_cb)
    if errors and not text:
        raise HTTPException(status_code=500, detail=" | ".join(errors))
    if not text:
        raise HTTPException(status_code=404, detail="No text could be extracted from the target.")

    scraped_text = f"\\n\\n--- Source: {url} ---\\n{text}"
    
    try:
        ai_response = invoke_ai_agent(scraped_text, user_prompt)
        
        clean_res = ai_response.strip()
        if clean_res.startswith("```json"):
            clean_res = clean_res[7:]
        if clean_res.startswith("```"):
            clean_res = clean_res[3:]
        if clean_res.endswith("```"):
            clean_res = clean_res[:-3]
            
        json_data = json.loads(clean_res.strip())
        return {"data": json_data}
    except json.JSONDecodeError:
        return {"raw_output": ai_response, "warning": "Failed to parse JSON object from AI output."}
    except Exception as e:
        logger.error(f"AI API Error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred with the AI API: {str(e)}")


@app.post("/api/scrape/custom", summary="Custom Scrape")
def scrape_custom(req: CustomScrapeRequest):
    """Scrape one or more URLs and return the AI-extracted result.
    
    Pass multiple URLs as a list: {"urls": ["https://wionews.com", "https://indiatoday.in"], "prompt": "..."}
    """
    DEFAULT_NEWS_PROMPT = (
        f"Analyze the provided web content.\n"
        f"CRITICAL RULE: You must ONLY extract information IF the content explicitly involves one or more of these specific Crisis Keywords:\n"
        f"{', '.join(CRISIS_KEYWORDS[:50])}...\n\n"
        f"If the page relates to any of these keywords, extract the core details of the event or crisis, "
        f"summarizing the who, what, when, where, and why.\n"
        f"If none of these crisis keywords apply, reply exactly with: "
        f"'No relevant crisis data found on this page based on the key attributes.'"
    )

    errors = []
    scraped_text = ""

    for idx, raw_url in enumerate(req.urls):
        target_url = raw_url.strip()
        if not target_url.startswith("http://") and not target_url.startswith("https://"):
            target_url = "https://" + target_url

        def error_cb(err, _url=target_url):
            errors.append(f"{_url}: {err}")

        text = scrape_website(target_url, use_browser=req.use_browser, error_callback=error_cb)
        if text:
            scraped_text += f"\n\n--- Source {idx+1}: {target_url} ---\n{text[:12000]}"

    if not scraped_text:
        raise HTTPException(
            status_code=500 if errors else 404,
            detail=" | ".join(errors) if errors else "No text could be extracted from any of the target URLs."
        )

    use_prompt = req.prompt if req.prompt.strip() else DEFAULT_NEWS_PROMPT

    try:
        ai_response = invoke_ai_agent(scraped_text, use_prompt)
        return {"sources_scraped": len(req.urls), "result": ai_response}
    except Exception as e:
        logger.error(f"AI API Error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred with the AI API: {str(e)}")


# ═══════════════════════════════════════════════════════════════════
#  GET ENDPOINTS — open these directly in your browser to get data
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/scrape/supply-chain", summary="Supply Chain Disruption [GET]")
def get_supply_chain(
    search_query: str = Query(default="Latest global supply chain disruptions 2026", description="Search query for DuckDuckGo"),
    use_browser: bool = Query(default=False, description="Use Selenium headless browser")
):
    """
    Open directly in browser:
    http://localhost:8000/api/scrape/supply-chain
    http://localhost:8000/api/scrape/supply-chain?search_query=shipping delays 2026
    """
    return scrape_supply_chain(SupplyChainRequest(search_query=search_query, use_browser=use_browser))


@app.get("/api/scrape/world-bank", summary="World Bank Automated [GET]")
def get_world_bank(
    use_browser: bool = Query(default=False, description="Use Selenium headless browser")
):
    """
    Open directly in browser:
    http://localhost:8000/api/scrape/world-bank
    """
    return scrape_world_bank(WorldBankRequest(use_browser=use_browser))


@app.get("/api/scrape/smart-resource", summary="Smart Resource Allocation AI [GET]")
def get_smart_resource(
    url: str = Query(default="https://data.worldbank.org/", description="Target URL to scrape"),
    allocation_objective: str = Query(default="Allocate a $100M development budget across regions to maximize socioeconomic impact.", description="Allocation goal"),
    use_browser: bool = Query(default=False, description="Use Selenium headless browser")
):
    """
    Open directly in browser:
    http://localhost:8000/api/scrape/smart-resource
    http://localhost:8000/api/scrape/smart-resource?url=https://data.worldbank.org&allocation_objective=Allocate $100M budget
    """
    return scrape_smart_resource(SmartResourceRequest(url=url, allocation_objective=allocation_objective, use_browser=use_browser))


@app.get("/api/scrape/custom", summary="Custom Scrape [GET]")
def get_custom_scrape(
    urls: str = Query(
        default="https://www.wionews.com/, https://www.indiatoday.in/",
        description="Comma-separated list of URLs to scrape"
    ),
    prompt: str = Query(
        default="",
        description="What should the AI extract? Leave blank to use the default crisis-keyword extraction prompt."
    ),
    use_browser: bool = Query(default=False, description="Use Selenium headless browser")
):
    """
    Open directly in browser (uses WION News + India Today by default):
    http://localhost:8000/api/scrape/custom
    http://localhost:8000/api/scrape/custom?urls=https://bbc.com,https://reuters.com&prompt=Summarize
    """
    url_list = [u.strip() for u in urls.split(",") if u.strip()]
    return scrape_custom(CustomScrapeRequest(urls=url_list, prompt=prompt, use_browser=use_browser))


# ─────────────────────────────────────────────
# DEDICATED NEWS ENDPOINT — scrapes WION + India Today by default
# ─────────────────────────────────────────────
@app.get("/api/scrape/news", summary="News Scraper — WION & India Today [GET]")
def get_news(
    urls: str = Query(
        default="https://www.wionews.com/, https://www.indiatoday.in/",
        description="Comma-separated news URLs (defaults to WION News + India Today)"
    ),
    prompt: str = Query(
        default="",
        description="Extraction prompt. Leave blank for default crisis-keyword analysis."
    ),
    use_browser: bool = Query(default=False, description="Use Selenium headless browser")
):
    """
    Dedicated news scraper. Defaults to WION News + India Today.
    
    Open directly in browser:
    http://localhost:8000/api/scrape/news
    http://localhost:8000/api/scrape/news?urls=https://ndtv.com,https://thehindu.com
    """
    url_list = [u.strip() for u in urls.split(",") if u.strip()]
    return scrape_custom(CustomScrapeRequest(urls=url_list, prompt=prompt, use_browser=use_browser))
