/**
 * Metrics visualization script
 * Loads JSONL metrics data and creates interactive Plotly charts
 */

let metricsData = [];
let chartsData = {};

/**
 * Fetch and parse JSONL metrics file
 */
async function loadMetrics() {
    try {
        const response = await fetch("../metrics/metrics.json");
        const text = await response.text();

        // Parse JSONL format
        metricsData = text
            .trim()
            .split("\n")
            .filter((line) => line.trim())
            .map((line) => JSON.parse(line));

        console.log(`Loaded ${metricsData.length} metrics entries`);
        processMetrics();
        updateCharts();
        updateStats();
    } catch (error) {
        console.error("Failed to load metrics:", error);
        document.body.innerHTML = `
            <div style="text-align: center; padding: 2rem;">
                <h2>📊 Metrics Not Available</h2>
                <p>Metrics file not found. Benchmarks will be generated on the first merge to master.</p>
                <p style="color: #666; margin-top: 1rem;">
                    ${error.message}
                </p>
            </div>
        `;
    }
}

/**
 * Process metrics data into chart format
 */
function processMetrics() {
    const timeMetrics = {};
    const throughputMetrics = {};
    const maxTimeMetrics = {};

    // Group metrics by scenario
    metricsData.forEach((entry) => {
        const timestamp = entry.timestamp;
        const revision = entry.revision || "unknown";

        Object.entries(entry.metrics).forEach(([key, [value, unit]]) => {
            if (key.endsWith("/time_ms")) {
                const scenario = key.replace("/time_ms", "");
                if (!timeMetrics[scenario]) {
                    timeMetrics[scenario] = { x: [], y: [] };
                }
                timeMetrics[scenario].x.push(timestamp);
                timeMetrics[scenario].y.push(value);
            } else if (key.endsWith("/throughput")) {
                const scenario = key.replace("/throughput", "");
                if (!throughputMetrics[scenario]) {
                    throughputMetrics[scenario] = { x: [], y: [] };
                }
                throughputMetrics[scenario].x.push(timestamp);
                throughputMetrics[scenario].y.push(value);
            } else if (key.endsWith("/max_ms")) {
                const scenario = key.replace("/max_ms", "");
                if (!maxTimeMetrics[scenario]) {
                    maxTimeMetrics[scenario] = { x: [], y: [] };
                }
                maxTimeMetrics[scenario].x.push(timestamp);
                maxTimeMetrics[scenario].y.push(value);
            }
        });
    });

    chartsData = {
        time: timeMetrics,
        throughput: throughputMetrics,
        maxTime: maxTimeMetrics,
    };
}

/**
 * Create Plotly trace from data
 */
function createTrace(name, data, normalize = false) {
    let y = [...data.y];

    if (normalize && y.length > 0) {
        const firstValue = y[0];
        y = y.map((v) => v / firstValue);
    }

    return {
        name: name,
        x: data.x.map((t) => new Date(t * 1000)),
        y: y,
        type: "scatter",
        mode: "lines+markers",
        hovertemplate: "<b>%{fullData.name}</b><br>Date: %{x|%Y-%m-%d %H:%M:%S}<br>Value: %{y:.2f}<extra></extra>",
    };
}

/**
 * Create comparison bar chart for latest metrics
 */
function createComparisonChart() {
    if (metricsData.length === 0) {
        return;
    }

    const latestEntry = metricsData[metricsData.length - 1];
    const metrics = latestEntry.metrics;

    const scenarios = {};

    Object.entries(metrics).forEach(([key, [value, unit]]) => {
        if (key.endsWith("/time_ms")) {
            const scenario = key.replace("/time_ms", "");
            if (!scenarios[scenario]) {
                scenarios[scenario] = {};
            }
            scenarios[scenario].time = value;
        }
        if (key.endsWith("/throughput")) {
            const scenario = key.replace("/throughput", "");
            if (!scenarios[scenario]) {
                scenarios[scenario] = {};
            }
            scenarios[scenario].throughput = value;
        }
    });

    const scenarioNames = Object.keys(scenarios).sort();
    const throughputs = scenarioNames.map((s) => scenarios[s].throughput || 0);

    const trace = {
        x: scenarioNames,
        y: throughputs,
        type: "bar",
        marker: { color: "#4ecdc4" },
        hovertemplate: "<b>%{x}</b><br>Throughput: %{y:.1f} img/s<extra></extra>",
    };

    const layout = {
        title: "Latest Performance by Scenario",
        xaxis: { title: "Scenario" },
        yaxis: { title: "Throughput (images/second)" },
        margin: { l: 50, r: 50, t: 50, b: 100 },
        xaxis: {
            tickangle: -45,
        },
        autosize: true,
    };

    Plotly.newPlot("comparisonChart", [trace], layout, { responsive: true });
}

/**
 * Update all charts
 */
function updateCharts() {
    const normalize = document.getElementById("normalizeCheckbox").checked;

    // Time Chart
    const timeTraces = Object.entries(chartsData.time).map(([name, data]) => createTrace(name, data, normalize));
    const timeLayout = {
        title: normalize ? "Processing Time (normalized to first value)" : "Processing Time",
        xaxis: { title: "Date" },
        yaxis: { title: "Time (ms)" },
        hovermode: "x unified",
        margin: { l: 50, r: 50, t: 50, b: 50 },
        autosize: true,
    };
    Plotly.newPlot("timeChart", timeTraces, timeLayout, { responsive: true });

    // Throughput Chart
    const throughputTraces = Object.entries(chartsData.throughput).map(([name, data]) =>
        createTrace(name, data, normalize)
    );
    const throughputLayout = {
        title: "Throughput Over Time",
        xaxis: { title: "Date" },
        yaxis: { title: "Images per Second" },
        hovermode: "x unified",
        margin: { l: 50, r: 50, t: 50, b: 50 },
        autosize: true,
    };
    Plotly.newPlot("throughputChart", throughputTraces, throughputLayout, { responsive: true });

    // Max Time Chart
    const maxTimeTraces = Object.entries(chartsData.maxTime).map(([name, data]) => createTrace(name, data, normalize));
    const maxTimeLayout = {
        title: "Maximum Processing Time",
        xaxis: { title: "Date" },
        yaxis: { title: "Time (ms)" },
        hovermode: "x unified",
        margin: { l: 50, r: 50, t: 50, b: 50 },
        autosize: true,
    };
    Plotly.newPlot("maxTimeChart", maxTimeTraces, maxTimeLayout, { responsive: true });

    // Comparison Chart
    createComparisonChart();
}

/**
 * Update stats display
 */
function updateStats() {
    const totalRuns = metricsData.length;
    document.getElementById("totalRuns").textContent = totalRuns;

    if (metricsData.length > 0) {
        const latest = metricsData[metricsData.length - 1];
        document.getElementById("latestCommit").textContent = latest.revision || "unknown";

        const lastDate = new Date(latest.timestamp * 1000);
        document.getElementById("lastUpdated").textContent = lastDate.toLocaleString();
    }
}

/**
 * Download metrics as JSON
 */
function downloadMetrics() {
    const dataStr = metricsData.map((d) => JSON.stringify(d)).join("\n");
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `metrics-${new Date().toISOString().split("T")[0]}.jsonl`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

/**
 * Handle window resize for responsive charts
 */
window.addEventListener("resize", () => {
    Plotly.Plots.resize("timeChart");
    Plotly.Plots.resize("throughputChart");
    Plotly.Plots.resize("maxTimeChart");
    Plotly.Plots.resize("comparisonChart");
});

// Load metrics when page loads
document.addEventListener("DOMContentLoaded", () => {
    loadMetrics();
});
