import datetime
from jinja2 import Template

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Risk Assessment Report - {{ project.project_name }}</title>
    <style>
        :root {
            --bg-base: #ffffff;
            --bg-card: #fafafa;
            --border-subtle: #eaeaea;
            --text-primary: #111111;
            --text-secondary: #666666;
            --accent: #8b5cf6;
            
            --risk-low: #10b981;
            --risk-warn: #f59e0b;
            --risk-high: #f43f5e;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-primary);
            line-height: 1.6;
            margin: 0;
            padding: 40px;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }

        h1, h2, h3, h4 {
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-top: 2em;
            margin-bottom: 0.5em;
        }

        h1 {
            font-size: 2.5rem;
            margin-top: 0;
            border-bottom: 2px solid var(--accent);
            padding-bottom: 10px;
            display: inline-block;
        }

        .meta-header {
            display: flex;
            justify-content: space-between;
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 40px;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 20px;
        }

        .section {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            page-break-inside: avoid;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }

        .metric-card {
            background: #fff;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }

        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid var(--border-subtle);
        }

        th {
            font-size: 0.85rem;
            text-transform: uppercase;
            color: var(--text-secondary);
            font-weight: 600;
            background: var(--bg-card);
        }

        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            color: #fff;
        }

        .badge-low { background: var(--risk-low); }
        .badge-warn { background: var(--risk-warn); }
        .badge-high { background: var(--risk-high); }

        .recommendation {
            margin-bottom: 16px;
            padding-bottom: 16px;
            border-bottom: 1px dashed var(--border-subtle);
        }
        .recommendation:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }

        .swot-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .swot-box {
            background: #fff;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 16px;
        }
        .swot-box h4 {
            margin-top: 0;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }

        @media print {
            body {
                background-color: white;
                padding: 0;
                max-width: 100%;
            }
            .section {
                border: none;
                padding: 0;
                background: transparent;
            }
            .metric-card {
                border: 1px solid #ccc;
            }
            /* Hide print button if present */
            .no-print {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="no-print" style="text-align: right; margin-bottom: 20px;">
        <button onclick="window.print()" style="background: var(--accent); color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600;">Print Report / Save PDF</button>
    </div>

    <h1>Comprehensive Assessment</h1>
    
    <div class="meta-header">
        <div>
            <strong>Project:</strong> {{ project.project_name }}<br>
            <strong>Industry:</strong> {{ project.industry_sector }}<br>
            <strong>Business Model:</strong> {{ project.business_model }}
        </div>
        <div style="text-align: right;">
            <strong>Date Generated:</strong> {{ date }}<br>
            <strong>Report ID:</strong> {{ project.project_id }}
        </div>
    </div>

    <h2>1. Executive Summary</h2>
    <div class="section grid-2">
        <div class="metric-card">
            <div class="metric-value" style="color: {{ 'var(--risk-high)' if analysis.failure_risk_score >= 60 else 'var(--risk-warn)' if analysis.failure_risk_score >= 35 else 'var(--risk-low)' }}">{{ analysis.failure_risk_score }}%</div>
            <div class="metric-label">Failure Risk Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color: {{ 'var(--risk-low)' if analysis.feasibility_score >= 65 else 'var(--risk-warn)' if analysis.feasibility_score >= 40 else 'var(--risk-high)' }}">{{ analysis.feasibility_score }}%</div>
            <div class="metric-label">Feasibility Score</div>
        </div>
    </div>
    <div class="section">
        <h3>Project Description</h3>
        <p>{{ project.description }}</p>
    </div>

    <h2>2. Risk Assessment</h2>
    <div class="section">
        <table>
            <thead>
                <tr>
                    <th>Risk Category</th>
                    <th>Score</th>
                    <th>Level</th>
                </tr>
            </thead>
            <tbody>
                {% for k, v in analysis.risk_breakdown.items() %}
                <tr>
                    <td>{{ k | replace('_', ' ') | title }}</td>
                    <td>{{ v }} / 100</td>
                    <td>
                        {% if v >= 60 %}
                            <span class="badge badge-high">High</span>
                        {% elif v >= 35 %}
                            <span class="badge badge-warn">Moderate</span>
                        {% else %}
                            <span class="badge badge-low">Low</span>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <h2>3. SWOT Analysis</h2>
    <div class="section swot-grid">
        <div class="swot-box">
            <h4 style="color: var(--risk-low);">Strengths</h4>
            <ul>
                {% for item in analysis.swot.strengths %}
                <li>{{ item }}</li>
                {% endfor %}
            </ul>
        </div>
        <div class="swot-box">
            <h4 style="color: var(--risk-warn);">Weaknesses</h4>
            <ul>
                {% for item in analysis.swot.weaknesses %}
                <li>{{ item }}</li>
                {% endfor %}
            </ul>
        </div>
        <div class="swot-box">
            <h4 style="color: var(--accent);">Opportunities</h4>
            <ul>
                {% for item in analysis.swot.opportunities %}
                <li>{{ item }}</li>
                {% endfor %}
            </ul>
        </div>
        <div class="swot-box">
            <h4 style="color: var(--risk-high);">Threats</h4>
            <ul>
                {% for item in analysis.swot.threats %}
                <li>{{ item }}</li>
                {% endfor %}
            </ul>
        </div>
    </div>

    {% if analysis.competitors %}
    <h2>4. Competition Assessment</h2>
    <div class="section">
        <table>
            <thead>
                <tr>
                    <th>Competitor Name</th>
                    <th>Threat Level</th>
                    <th>Key Advantage</th>
                </tr>
            </thead>
            <tbody>
                {% for comp in analysis.competitors %}
                <tr>
                    <td><strong>{{ comp.name }}</strong></td>
                    <td>
                        {% if comp.threat_level == 'High' %}
                            <span class="badge badge-high">High</span>
                        {% elif comp.threat_level == 'Medium' %}
                            <span class="badge badge-warn">Medium</span>
                        {% else %}
                            <span class="badge badge-low">Low</span>
                        {% endif %}
                    </td>
                    <td>{{ comp.key_advantage }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

    {% if strategy %}
    <h2>5. Strategic Recommendations</h2>
    <div class="section">
        {% for rec in strategy.recommendations %}
        <div class="recommendation">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <h4 style="margin-top: 0; color: var(--accent);">{{ rec.title }}</h4>
                <span class="badge {{ 'badge-high' if rec.priority == 'Critical' else 'badge-warn' if rec.priority == 'High' else 'badge-low' }}">{{ rec.priority }}</span>
            </div>
            <p>{{ rec.description }}</p>
            <p><strong>Action:</strong> {{ rec.actionable_step }}</p>
        </div>
        {% endfor %}
    </div>

    <h2>6. Risk Mitigation Matrix</h2>
    <div class="section">
        <table>
            <thead>
                <tr>
                    <th>Risk Factor</th>
                    <th>Mitigation Strategy</th>
                    <th>Timeframe</th>
                </tr>
            </thead>
            <tbody>
                {% for risk in strategy.risk_mitigations %}
                <tr>
                    <td><strong>{{ risk.risk_factor }}</strong></td>
                    <td>{{ risk.mitigation_strategy }}</td>
                    <td>{{ risk.timeframe }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

    <div style="margin-top: 60px; text-align: center; color: var(--text-secondary); font-size: 0.85rem; border-top: 1px solid var(--border-subtle); padding-top: 20px;">
        <p>Smart Failure Detection &bull; Automated Assessment Report</p>
        <p><em>Disclaimer: Scores and analysis reflect deterministic multi-factor modeling and strategic AI recommendations. They should be used to inform, not replace, human due diligence.</em></p>
    </div>
</body>
</html>
"""

def generate_report(project: dict, analysis: dict, strategy: dict) -> str:
    template = Template(REPORT_TEMPLATE)
    return template.render(
        project=project,
        analysis=analysis,
        strategy=strategy,
        date=datetime.datetime.now().strftime("%B %d, %Y")
    )
