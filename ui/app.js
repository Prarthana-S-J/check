const API_BASE = window.location.origin;

const tripDropdown = document.getElementById('trip-dropdown');
const btnEvaluate = document.getElementById('btn-evaluate');
const mapFrame = document.getElementById('map-frame');
const logConsole = document.getElementById('log-console');

const valHmmAcc = document.getElementById('val-hmm-acc');
const valClfAcc = document.getElementById('val-clf-acc');
const accGain = document.getElementById('acc-gain');

const valHmmCont = document.getElementById('val-hmm-cont');
const valClfCont = document.getElementById('val-clf-cont');
const contGain = document.getElementById('cont-gain');

const valMissRate = document.getElementById('val-miss-rate');
const valObsCount = document.getElementById('val-obs-count');
const valLatency = document.getElementById('val-latency');

function appendLog(text, type = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    const timestamp = new Date().toLocaleTimeString();
    entry.textContent = `[${timestamp}] ${text}`;
    logConsole.appendChild(entry);
    logConsole.scrollTop = logConsole.scrollHeight;
}

// Fetch available trips
async function loadTrips() {
    try {
        appendLog('Fetching catalog of reserved evaluation trips from server...');
        const res = await fetch(`${API_BASE}/api/trips`);
        if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
        const data = await res.json();
        
        tripDropdown.innerHTML = '';
        data.trips.forEach((t, i) => {
            const opt = document.createElement('option');
            opt.value = t.trip_id;
            opt.textContent = `Trip ${t.trip_id} (${t.observations} GPS Points, ${t.road_arcs} Arcs)`;
            if (i === 0) opt.selected = true;
            tripDropdown.appendChild(opt);
        });

        btnEvaluate.disabled = false;
        appendLog(`Loaded ${data.trips.length} reserved evaluation trips. Ready for evaluation.`, 'success');
        
        // Auto trigger evaluation for first trip
        evaluateSelectedTrip();
    } catch (err) {
        appendLog(`Failed to connect to backend: ${err.message}`, 'warn');
    }
}

// Evaluate trip
async function evaluateSelectedTrip() {
    const tripId = tripDropdown.value;
    if (!tripId) return;

    btnEvaluate.disabled = true;
    btnEvaluate.innerHTML = '<span>&#9203;</span> Processing Inference...';
    appendLog(`>>> Dispatching evaluation request: POST /api/evaluate/${tripId}`);

    try {
        const t0 = performance.now();
        const res = await fetch(`${API_BASE}/api/evaluate/${tripId}`, {
            method: 'POST'
        });

        if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
        const result = await res.json();
        const roundTrip = ((performance.now() - t0) / 1000).toFixed(2);

        // Update UI Metrics
        valHmmAcc.textContent = `${(result.hmm_top1_accuracy * 100).toFixed(2)}%`;
        valClfAcc.textContent = `Clf: ${(result.clf_top1_accuracy * 100).toFixed(2)}%`;
        accGain.textContent = `${result.accuracy_gain_pct >= 0 ? '+' : ''}${result.accuracy_gain_pct.toFixed(2)}%`;

        valHmmCont.textContent = `${(result.hmm_route_continuity * 100).toFixed(2)}%`;
        valClfCont.textContent = `Clf: ${(result.clf_route_continuity * 100).toFixed(2)}%`;
        contGain.textContent = `${result.continuity_gain_pct >= 0 ? '+' : ''}${result.continuity_gain_pct.toFixed(2)}%`;

        valMissRate.textContent = `${(result.candidate_miss_rate * 100).toFixed(2)}%`;
        valObsCount.textContent = `${result.observations_count} Obs`;
        valLatency.textContent = `${result.processing_time_s}s`;

        // Update Map
        mapFrame.src = `${API_BASE}${result.map_url}?t=${Date.now()}`;

        appendLog(`<<< HTTP 200 OK | Trip: ${tripId} | Top-1 Acc: ${(result.hmm_top1_accuracy * 100).toFixed(2)}% | Continuity: ${(result.hmm_route_continuity * 100).toFixed(2)}% | Server: ${result.processing_time_s}s | Round-trip: ${roundTrip}s`, 'success');
        appendLog(`Reconstructed ${result.total_predicted_arcs} road segments. Interactive Folium map updated.`);
    } catch (err) {
        appendLog(`Evaluation request failed: ${err.message}`, 'warn');
    } finally {
        btnEvaluate.disabled = false;
        btnEvaluate.innerHTML = '<span>&#9654;</span> Evaluate Selected Trip';
    }
}

btnEvaluate.addEventListener('click', evaluateSelectedTrip);
tripDropdown.addEventListener('change', evaluateSelectedTrip);

// Start
loadTrips();

