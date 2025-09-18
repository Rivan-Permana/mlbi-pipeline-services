import os
import io
import json
import time
import uuid
from datetime import datetime
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google.cloud import storage, firestore
from litellm import completion
import pandasai as pai
from pandasai import SmartDataframe, SmartDatalake
from pandasai_litellm.litellm import LiteLLM
from pandasai.core.response.dataframe import DataFrameResponse

# Initialize FastAPI app
app = FastAPI(title="ML BI Pipeline API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost:5173", "https://convoinsight.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
PROJECT_ID = os.getenv("PROJECT_ID")
GCS_BUCKET = os.getenv("GCS_BUCKET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize clients
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)
db = firestore.Client()

# Pydantic models
class QueryRequest(BaseModel):
    domain: str
    prompt: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    session_id: str
    response: str
    chart_url: Optional[str] = None
    execution_time: float

def get_content(r):
    """Extract content from LLM response"""
    try:
        msg = r.choices[0].message
        return msg["content"] if isinstance(msg, dict) else msg.content
    except Exception:
        pass

    if isinstance(r, dict):
        return r.get("choices", [{}])[0].get("message", {}).get("content", "")

    try:
        chunks = []
        for ev in r:
            delta = getattr(ev.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                chunks.append(delta.content)
        return "".join(chunks)
    except Exception:
        return str(r)

def upload_to_gcs(file_path: str, destination_blob_name: str):
    """Upload file to Google Cloud Storage"""
    try:
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
        return f"gs://{GCS_BUCKET}/{destination_blob_name}"
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return None

def save_to_firestore(session_id: str, domain: str, prompt: str, response: str,
                     execution_time: float, chart_url: Optional[str] = None):
    """Save chat history to Firestore"""
    try:
        doc_ref = db.collection("chat_history").document(session_id)
        doc_ref.set({
            "domain": domain,
            "prompt": prompt,
            "response": response,
            "chart_url": chart_url,
            "execution_time": execution_time,
            "timestamp": datetime.utcnow(),
        })
    except Exception as e:
        print(f"Error saving to Firestore: {e}")

@app.post("/upload_datasets/{domain}")
async def upload_datasets(domain: str, files: List[UploadFile] = File(...)):
    """Upload CSV datasets for a domain"""
    try:
        uploaded_files = []

        for file in files:
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a CSV file")

            # Save file locally first
            local_path = f"/tmp/{file.filename}"
            with open(local_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Upload to GCS
            gcs_path = f"datasets/{domain}/{file.filename}"
            gcs_url = upload_to_gcs(local_path, gcs_path)

            if gcs_url:
                uploaded_files.append({
                    "filename": file.filename,
                    "gcs_path": gcs_url
                })

            # Clean up local file
            os.remove(local_path)

        return {"message": f"Uploaded {len(uploaded_files)} files for domain {domain}",
                "files": uploaded_files}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process ML BI Pipeline query"""
    start_time = time.time()

    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Download datasets from GCS
        datasets_path = f"datasets/{request.domain}/"
        blobs = list(bucket.list_blobs(prefix=datasets_path))

        if not blobs:
            raise HTTPException(status_code=404, detail=f"No datasets found for domain {request.domain}")

        # Download and load datasets
        dfs = {}
        data_info = {}
        data_describe = {}

        for blob in blobs:
            if blob.name.endswith('.csv'):
                filename = os.path.basename(blob.name)
                local_path = f"/tmp/{filename}"
                blob.download_to_filename(local_path)

                # Load dataframe
                df = pd.read_csv(local_path, sep='|')
                dfs[filename] = df

                # Get info and describe
                buf = io.StringIO()
                df.info(buf=buf)
                data_info[filename] = buf.getvalue()
                data_describe[filename] = df.describe(include='all')

                # Clean up
                os.remove(local_path)

        # Set API key
        api_key = GEMINI_API_KEY or OPENAI_API_KEY
        if not api_key:
            raise HTTPException(status_code=500, detail="No API key configured")

        # Orchestrate LLMs
        orchestrator_response = completion(
            model="gemini/gemini-2.5-pro",
            messages=[
                {"role": "system", "content": """
                You are the Orchestrator.

                15 instructions you need to follow as the orchestrator:
                1. Think step by step.
                2. You orchestrate 3 LLM PandasAI Agents for business data analysis.
                3. The 3 agents are: Data Manipulator, Data Visualizer, Data Analyser.
                4. You will emit a specific prompt for each of those 3 agents.
                5. Each prompt is a numbered, step-by-step instruction set.
                6. Prompts must be clear, detailed, and complete to avoid ambiguity.
                7. The number of steps may differ per agent.
                8. Example user tasks include:
                8a. What is my revenue this week vs last week?
                8b. Why did my revenue drop this week?
                8c. Any surprises in revenue this month?
                8d. Notable trends this month?
                8e. Correlation between revenue and bounces?
                8f. Is this conversion rate normal for this time of year?
                9. Reason strictly from the user-provided data.
                10. Convert a short business question into three specialist prompts.
                11. If a currency is not explicitly stated assume its in Indonesian Rupiah.
                13. All specialists operate in Python using PandasAI SmartDataframe as `sdf` (backed by pandas `df`).
                14. Return STRICT JSON with keys: manipulator_prompt, visualizer_prompt, analyzer_prompt, compiler_instruction.
                15. Each value must be a **single line** string. No extra keys, no prose, no markdown/code fences.

                6 instructions for data manipulator prompt creation:
                1. Enforce data hygiene before analysis.
                1a. Parse dates to pandas datetime, create explicit period columns (day/week/month).
                1b. Set consistent dtypes for numeric fields; strip/normalize categorical labels; standardize currency units if present.
                1c. Handle missing values: impute or drop **only when necessary**; keep legitimate zeros.
                2. Mind the term like m0 and m1 which means month 0 and 1 and any other similar terms used to decide if something is the former or later, in this case the m0 is the previous month and the m1 is the current or the next month.
                3. Mind each of the datasets name.
                4. Produce exactly the minimal, analysis-ready dataframe(s) needed for the user question, with stable, well-named columns.
                5. Include the percentage version of the raw value on the column that you think is appropriate to include.
                6. End by returning only: result = {"type":"dataframe","value": <THE_FINAL_DATAFRAME>}

                15 instructions for data visualizer prompt creation:
                1. Produce exactly ONE interactive visualization (a Plotly diagram or an HTML table) per request.
                2. Choose the best form of visualization based on the user's question. Use a Plotly diagram for trends and comparisons; use a styled HTML table for ranked lists or data with percentages.
                3. For Plotly diagrams: Prevent overlaps by rotating axis ticks ≤45°, wrapping long labels, ensuring adequate margins, and placing the legend outside the plot area.
                4. For Plotly diagrams: Apply insight-first formatting: include a clear title and subtitle, label axes with units, use thousands separators, and configure a rich hover-over.
                5. Aggregate data to a sensible granularity (e.g., day, week, or month) and cap extreme outliers for readability (noting this in the subtitle).
                6. For Plotly diagrams: To ensure high contrast, instruct the agent to use a truncated monochromatic colorscale by skipping the lightest 25% of a standard scale (e.g., 'Blues only').
                7. The prompt must specify how to truncate the scale, for example: "Create a custom colorscale by sampling 'Blues' from 0.25 to 1.0." The gradient must map to data values (lighter for low, darker for high).
                8. For Plotly diagrams: Use a bar chart, grouped bar chart, or line chart.
                9. If a table visualization is chosen, instruct the agent to use the Pandas Styler object to generate the final HTML, not Plotly.
                10. The prompt must specify using the Styler.bar() method only on columns that represent share-of-total percentages and only when the column total ≈ 100%.
                11. Output Python code only (no prose/comments/markdown). Import os and datetime. Build a directory and a run-scoped timestamped filename using a run ID stored in globals.
                12. Write the file exactly once using an atomic lock to avoid duplicates across retries.
                13. Ensure file_path is a plain Python string and do not print/return anything else.
                14. The last line of code MUST be exactly: result = {"type": "string", "value": file_path}
                15. DO NOT return the raw HTML string in the value field.

                3 instructions for data analyzer prompt creation:
                1. Write like you're speaking to a person; be concise and insight-driven.
                2. Quantify where possible (deltas, % contributions, time windows); reference the exact columns/filters used.
                3. Return only: result = {"type":"string","value":"<3–6 crisp bullets or 2 short paragraphs of insights>"}

                34 instructions for response compiler system content creation:
                1. Brevity: ≤180 words; bullets preferred; no code blocks, no JSON, no screenshots.
                2. Lead with the answer: 1–2 sentence "Bottom line" with main number, time window, and delta.
                3. Quantified drivers: top 3 with magnitude, direction, and approx contribution (absolute and % where possible).
                4. Next actions: 2–4 prioritized, concrete actions with expected impact/rationale.
                5. Confidence & caveats: one short line on data quality/assumptions/gaps; include Confidence: High/Medium/Low.
                6. Minimal tables: ≤1 table only if essential (≤5×3); otherwise avoid tables.
                7. No repetition: do not restate agent text; synthesize it.
                8. Do not try to show images; if a chart exists, mention the HTML path only.
                9. Always include units/currency and exact comparison window (e.g., "Aug 2025 vs Jul 2025", "W34 vs W33").
                10. Show both absolute and % change where sensible (e.g., "+$120k (+8.4%)").
                11. Round smartly (money to nearest K unless < $10k; rates 1–2 decimals).
                12. If any agent fails or data is incomplete, still produce the best insight; mark gaps in Caveats and adjust Confidence.
                13. If the user asks "how much/which/why," the first sentence must provide the number/entity/reason.
                14. Exact compiler_instruction template the orchestrator should emit (single line; steps separated by ';'):
                15. Read the user prompt, data_info, and all three agent responses;
                16. Compute the direct answer including the main number and compare period;
                17. Identify the top 3 quantified drivers with direction and contribution;
                18. Draft 'Bottom line' in 1–2 sentences answering plainly;
                19. List 2–4 prioritized Next actions with expected impact;
                20. Add a one-line Caveats with Confidence and any gaps;
                21. Keep ≤180 words, use bullets, avoid tables unless ≤5×3 and essential;
                22. Include units, absolute and % deltas, and explicit dates;
                23. Do not repeat agent text verbatim or include code/JSON.
                24. Format hint (shape, not literal): 24a. Bottom line – <answer with number + timeframe>. 24b. Drivers – <A: +X (≈Y%); B: −X (≈Y%); C: ±X (≈Y%)>. 24c. Next actions – 1) <action>; 2) <action>; 3) <action>. 24d. Caveats – <one line>. Confidence: <High/Medium/Low>.
                25. compiler_instruction must contain clear, step-by-step instructions to assemble the final response.
                26. The final response must be decision-ready and insight-first, not raw data.
                27. The compiler_instruction is used as the compiler LLM's system content.
                28. Compiler user content will be: f"User Prompt:{user_prompt}. \nData Info:{data_info}. \nData Describe:{data_describe}. \nData Manipulator Response:{data_manipulator_response}. \nData Visualizer Response:{data_visualizer_response}. \nData Analyzer Response:{data_analyzer_response}".
                29. `data_info` is a string from `df.info()`.
                30. `data_manipulator_response` is a PandasAI DataFrameResponse.
                31. `data_visualizer_response` is a **file path to an HTML** inside `{"type":"string","value": ...}`. The `value` MUST be a plain Python string containing the path.
                32. `data_analyzer_response` is a PandasAI StringResponse.
                33. Your goal in `compiler_instruction` is to force brevity, decisions, and insights.
                34. The compiler must NOT echo raw dataframes, code, or long tables; it opens with the business answer, quantifies drivers, and closes with next actions.
                """},
                {"role": "user", "content": f"User Prompt: {request.prompt} \nDatasets Domain name: {request.domain}. \ndf.info of each dfs key(file name)-value pair:\n{data_info}. \df.describe of each dfs key(file name)-value pair:\n{data_describe}."}
            ],
            seed=1,
            stream=False,
            verbosity="low",
            drop_params=True,
            reasoning_effort="high",
        )

        orchestrator_content = get_content(orchestrator_response)

        try:
            spec = json.loads(orchestrator_content)
        except json.JSONDecodeError:
            start = orchestrator_content.find("{")
            end = orchestrator_content.rfind("}")
            spec = json.loads(orchestrator_content[start:end+1])

        manipulator_prompt = spec["manipulator_prompt"]
        visualizer_prompt = spec["visualizer_prompt"]
        analyzer_prompt = spec["analyzer_prompt"]
        compiler_instruction = spec["compiler_instruction"]

        # Setup LLM
        llm = LiteLLM(model="gemini/gemini-2.5-pro", api_key=api_key)
        pai.config.set({"llm": llm})

        # Data Manipulator
        data_manipulator = SmartDatalake(
            list(dfs.values()),
            config={
                "llm": llm,
                "seed": 1,
                "stream": False,
                "verbosity": "low",
                "drop_params": True,
                "save_charts": False,
                "open_charts": False,
                "conversational": False,
                "enforce_privacy": True,
                "reasoning_effort": "high",
            }
        )
        data_manipulator_response = data_manipulator.chat(manipulator_prompt)

        # Get processed dataframe
        if isinstance(data_manipulator_response, DataFrameResponse):
            df_processed = data_manipulator_response.value
        else:
            df_processed = data_manipulator_response

        # Data Visualizer
        data_visualizer = SmartDataframe(
            df_processed,
            config={
                "llm": llm,
                "seed": 1,
                "stream": False,
                "verbosity": "low",
                "drop_params": True,
                "save_charts": False,
                "open_charts": False,
                "conversational": False,
                "enforce_privacy": True,
                "reasoning_effort": "high",
            }
        )

        # Set global run ID for file naming
        import datetime as _dt
        run_id = _dt.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        globals()["_RUN_ID"] = run_id

        data_visualizer_response = data_visualizer.chat(visualizer_prompt)

        # Data Analyzer
        data_analyzer = SmartDataframe(
            df_processed,
            config={
                "llm": llm,
                "seed": 1,
                "stream": False,
                "verbosity": "low",
                "drop_params": True,
                "save_charts": False,
                "open_charts": False,
                "conversational": True,
                "enforce_privacy": False,
                "reasoning_effort": "high",
            }
        )
        data_analyzer_response = data_analyzer.chat(analyzer_prompt)

        # Response Compiler
        final_response = completion(
            model="gemini/gemini-2.5-pro",
            messages=[
                {"role": "system", "content": compiler_instruction},
                {"role": "user", "content": f"User Prompt:{request.prompt}. \nDatasets Domain name: {request.domain}. \ndf.info of each dfs key(file name)-value pair:\n{data_info}. \df.describe of each dfs key(file name)-value pair:\n{data_describe}. \nData Visualizer Response:{data_visualizer_response.value}. \nData Analyzer Response:{data_analyzer_response}."},
            ],
            seed=1,
            stream=False,
            verbosity="medium",
            drop_params=True,
            reasoning_effort="high",
        )
        final_content = get_content(final_response)

        # Upload chart to GCS if it exists
        chart_url = None
        if hasattr(data_visualizer_response, 'value') and data_visualizer_response.value:
            chart_path = data_visualizer_response.value
            if os.path.exists(chart_path):
                gcs_chart_path = f"charts/{request.domain}/{session_id}_{run_id}.html"
                chart_url = upload_to_gcs(chart_path, gcs_chart_path)
                os.remove(chart_path)  # Clean up local file

        # Calculate execution time
        execution_time = time.time() - start_time

        # Save to Firestore
        save_to_firestore(session_id, request.domain, request.prompt, final_content, execution_time, chart_url)

        return QueryResponse(
            session_id=session_id,
            response=final_content,
            chart_url=chart_url,
            execution_time=execution_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        doc_ref = db.collection("chat_history").document(session_id)
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/domains/{domain}/datasets")
async def list_domain_datasets(domain: str):
    """List all datasets for a domain"""
    try:
        datasets_path = f"datasets/{domain}/"
        blobs = list(bucket.list_blobs(prefix=datasets_path))

        datasets = []
        for blob in blobs:
            if blob.name.endswith('.csv'):
                datasets.append({
                    "filename": os.path.basename(blob.name),
                    "gcs_path": f"gs://{GCS_BUCKET}/{blob.name}",
                    "size": blob.size,
                    "updated": blob.updated.isoformat() if blob.updated else None
                })

        return {"domain": domain, "datasets": datasets}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
