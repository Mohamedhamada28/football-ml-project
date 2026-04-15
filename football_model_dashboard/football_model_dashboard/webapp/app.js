const state = {
  dashboard: null,
  lastPositionPrediction: null,
  lastDatasetResult: null,
};
console.log('app.js loaded');

const FIELD_LABELS = {
  Nation: "Nation",
  Squad: "Squad",
  Comp: "League",
  Pos: "Position",
  season: "Season",
  Age: "Age",
  MP: "Matches Played",
  Starts: "Starts",
  Min: "Minutes",
  "90s": "90s",
  Gls: "Goals",
  Ast: "Assists",
  PK: "Penalty Goals",
  PKatt: "Penalty Attempts",
  CrdY: "Yellow Cards",
  CrdR: "Red Cards",
  xG: "xG",
  npxG: "npxG",
  xAG: "xAG",
  PrgC: "Progressive Carries",
  PrgP: "Progressive Passes",
  PrgR: "Progressive Receptions",
};

const POSITION_FORM_GROUPS = [
  {
    title: "Profile & Context",
    fields: ["Nation", "Squad", "Comp", "season"],
  },
  {
    title: "Playing Time",
    fields: ["Age", "MP", "Starts", "Min", "90s"],
  },
  {
    title: "Role Signals",
    fields: ["Gls", "Ast", "PK", "PKatt", "CrdY", "CrdR", "xG", "npxG", "xAG", "PrgC", "PrgP", "PrgR"],
  },
];

const GOALS_FORM_GROUPS = [
  {
    title: "Profile & Context",
    fields: ["Nation", "Squad", "Comp", "Pos", "season"],
  },
  {
    title: "Playing Time",
    fields: ["Age", "MP", "Starts", "Min", "90s"],
  },
  {
    title: "Creation & Progression",
    fields: ["Ast", "PK", "PKatt", "CrdY", "CrdR", "xG", "npxG", "xAG", "PrgC", "PrgP", "PrgR"],
  },
];

document.addEventListener("DOMContentLoaded", () => {
  initializeDashboard()
    .catch((error) => {
      renderLoadError(error);
    })
    .finally(() => {
      attachDatasetFormHandler();
    });
});

function attachDatasetFormHandler() {
  const form = document.getElementById("dataset-form");
  if (!form) {
    console.error("Dataset form not found - cannot attach handler");
    return;
  }

  form.onsubmit = (event) => {
    console.log("=== FORM SUBMIT EVENT FIRED ===");
    event.preventDefault();
    event.stopPropagation();
    console.log("Default prevented, calling handleDatasetSubmit");
    handleDatasetSubmit(event);
    return false;
  };
  
  console.log("Dataset form onsubmit handler attached successfully");
}

async function initializeDashboard() {
  const dashboard = await fetchJSON("/api/dashboard");
  state.dashboard = dashboard;

  renderHero(dashboard.hero);
  renderKpis(dashboard.kpis);
  renderClassificationChart(dashboard.classificationComparison);
  renderRegressionChart(dashboard.regressionComparison, dashboard.baselineComparison);
  renderModelExamples(dashboard.modelExamples);
  renderFindings(dashboard.findings);
  renderClassBreakdown(dashboard.classBreakdown);
  renderMatrices(dashboard.confusionMatrices);
  renderFeatureHighlights(dashboard.featureHighlights);
  renderPredictor(dashboard.predictor);
  renderAwardRadar(dashboard.awardRadar);
  renderDatasetLab(dashboard.predictor.schema);
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed.");
  }
  return payload;
}

function renderLoadError(error) {
  const message = `
    <div class="loading-card">
      <strong>Dashboard failed to load.</strong>
      <p>${error.message}</p>
    </div>
  `;
  ["kpi-grid", "classification-chart", "regression-chart", "findings-grid"].forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      element.innerHTML = message;
    }
  });

  ["dataset-summary", "dataset-top10", "dataset-league-top5"].forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      element.innerHTML = message;
    }
  });
}

function renderHero(hero) {
  document.getElementById("hero-title").textContent = hero.title;
  document.getElementById("hero-subtitle").textContent = hero.subtitle;

  const heroMeta = document.getElementById("hero-meta");
  const metaRows = [
    { label: "Rows in shared base", value: hero.meta.combined_rows },
    { label: "Rows in detailed 2024/25", value: hero.meta.detailed_rows },
    { label: "Inputs for live classifier", value: hero.meta.combined_features },
    { label: "Inputs in detailed comparison model", value: hero.meta.detailed_features },
  ];
  heroMeta.innerHTML = metaRows
    .map(
      (item) => `
        <div class="hero-meta-item">
          <span>${item.label}</span>
          <strong>${item.value}</strong>
        </div>
      `
    )
    .join("");
}

function renderKpis(kpis) {
  const container = document.getElementById("kpi-grid");
  container.innerHTML = kpis
    .map(
      (item) => `
        <article class="kpi-card ${item.tone}">
          <div class="card-kicker">${item.label}</div>
          <div class="kpi-value">${item.value}${item.suffix}</div>
          <div class="kpi-detail">${item.detail}</div>
        </article>
      `
    )
    .join("");
}

function renderClassificationChart(rows) {
  const container = document.getElementById("classification-chart");
  const groups = groupBy(rows, "experimentLabel");

  container.innerHTML = Object.entries(groups)
    .map(([experimentLabel, items]) => {
      const bars = items
        .map(
          (item) => `
            <div class="bar-row">
              <div class="bar-label">${item.model}</div>
              <div>
                <div class="bar-track">
                  <div class="bar-fill" style="width: ${item.accuracy}%"></div>
                </div>
                <div class="metrics-inline">
                  <span class="metric-chip">Accuracy ${item.accuracy}%</span>
                  <span class="metric-chip">Macro-F1 ${item.macroF1}%</span>
                  <span class="metric-chip">MF recall ${item.midfielderRecall}%</span>
                </div>
              </div>
              <div class="bar-value">${item.accuracy}%</div>
            </div>
          `
        )
        .join("");

      return `
        <div class="chart-group">
          <div class="chart-group-title">${experimentLabel}</div>
          ${bars}
        </div>
      `;
    })
    .join("");
}

function renderRegressionChart(rows, baseline) {
  const container = document.getElementById("regression-chart");
  const maxR2 = Math.max(...rows.map((row) => row.r2), 1);

  container.innerHTML = rows
    .map((item, index) => {
      const width = Math.max((item.r2 / maxR2) * 100, 0);
      return `
        <div class="chart-group">
          <div class="bar-row">
            <div class="bar-label">${item.model}</div>
            <div>
              <div class="bar-track">
                <div class="bar-fill ${index === 0 ? "" : "alt"}" style="width: ${width}%"></div>
              </div>
              <div class="metrics-inline">
                <span class="metric-chip">R2 ${item.r2}</span>
                <span class="metric-chip">MAE ${item.mae}</span>
                <span class="metric-chip">RMSE ${item.rmse}</span>
                <span class="metric-chip">MSE ${item.mse}</span>
              </div>
            </div>
            <div class="bar-value">${item.r2}</div>
          </div>
        </div>
      `;
    })
    .join("");

  document.getElementById("baseline-callout").innerHTML = `
    <strong>Baseline vs refresh.</strong>
    Notebook RF holdout accuracy was ${baseline.classification_holdout_baseline}%. The shared-base winner adds ${baseline.combined_improvement} pts, and the detailed winner adds ${baseline.detailed_improvement} pts.
    The notebook linear regression baseline landed at R2 ${baseline.regression_baseline_r2} with MSE ${baseline.regression_baseline_mse}; the exported best regressor trims MSE by ${Math.abs(baseline.regression_mse_delta).toFixed(3)}.
  `;
}

function renderModelExamples(modelExamples) {
  document.getElementById("position-examples").innerHTML = `
    <div class="example-section">
      <div class="note-chip">Examples use ${modelExamples.modelsUsed.position}</div>
      <div class="example-title">Examples the position model nailed</div>
      <div class="example-grid">
        ${modelExamples.position.accurate.map(renderPositionExampleCard).join("")}
      </div>
    </div>
    <div class="example-section">
      <div class="example-title">Examples where roles overlapped</div>
      <div class="example-grid">
        ${modelExamples.position.misses.map(renderPositionExampleCard).join("")}
      </div>
    </div>
  `;

  document.getElementById("goal-examples").innerHTML = `
    <div class="example-section">
      <div class="note-chip">Examples use ${modelExamples.modelsUsed.goals}</div>
      <div class="example-title">Closest goal predictions</div>
      <div class="example-grid">
        ${modelExamples.goals.accurate.map(renderGoalExampleCard).join("")}
      </div>
    </div>
    <div class="example-section">
      <div class="example-title">Biggest goal misses</div>
      <div class="example-grid">
        ${modelExamples.goals.misses.map(renderGoalExampleCard).join("")}
      </div>
    </div>
  `;
}

function renderPositionExampleCard(example) {
  const correct = example.actual === example.predicted;
  return `
    <article class="example-card">
      <div class="example-top">
        <div>
          <strong>${example.player}</strong>
          <span>${example.squad}</span>
        </div>
        <span class="example-badge ${correct ? "good" : "warn"}">
          ${example.actual} → ${example.predicted}
        </span>
      </div>
      <div class="metrics-inline">
        <span class="metric-chip">Confidence ${example.confidence}%</span>
        <span class="metric-chip">G+A ${example.goalContrib}</span>
        <span class="metric-chip">PrgP ${example.progressivePasses}</span>
        <span class="metric-chip">PrgR ${example.progressiveReceptions}</span>
      </div>
    </article>
  `;
}

function renderGoalExampleCard(example) {
  return `
    <article class="example-card">
      <div class="example-top">
        <div>
          <strong>${example.player}</strong>
          <span>${example.squad}</span>
        </div>
        <span class="example-badge">
          ${example.position} / ${example.predictedPosition}
        </span>
      </div>
      <div class="metrics-inline">
        <span class="metric-chip">Actual ${example.actualGoals}</span>
        <span class="metric-chip">Pred ${example.predictedGoals}</span>
        <span class="metric-chip">Error ${example.goalError}</span>
        <span class="metric-chip">Conf ${example.positionConfidence}%</span>
      </div>
    </article>
  `;
}

function renderFindings(findings) {
  const container = document.getElementById("findings-grid");
  container.innerHTML = findings
    .map(
      (finding) => `
        <article class="finding-card">
          <div class="card-kicker">${finding.eyebrow}</div>
          <div class="finding-value">${finding.value}</div>
          <h3>${finding.title}</h3>
          <p>${finding.body}</p>
        </article>
      `
    )
    .join("");
}

function renderAwardRadar(awardRadar) {
  document.getElementById("award-subtitle").textContent = awardRadar.subtitle;
  document.getElementById("award-methodology").textContent = awardRadar.methodology;
  document.getElementById("award-model-stack").innerHTML = `
    <span class="note-chip">${awardRadar.modelsUsed.position}</span>
    <span class="note-chip">${awardRadar.modelsUsed.goals}</span>
  `;
  document.getElementById("award-how").innerHTML = awardRadar.howItWorks
    .map((item) => `<div class="how-item">${item}</div>`)
    .join("");

  document.getElementById("award-insights").innerHTML = awardRadar.insights
    .map(
      (insight) => `
        <article class="award-insight-card">
          <div class="card-kicker">${insight.label}</div>
          <strong>${insight.value}</strong>
          <p>${insight.detail}</p>
        </article>
      `
    )
    .join("");

  document.getElementById("award-candidates").innerHTML = awardRadar.candidates
    .map(
      (candidate, index) => `
        <article class="candidate-card ${index === 0 ? "featured" : ""}">
          <div class="candidate-rank">#${index + 1}</div>
          <div class="candidate-main">
            <div>
              <h3>${candidate.player}</h3>
              <p>${candidate.squad} · ${candidate.league} · ${candidate.positionLabel}</p>
            </div>
            <div class="candidate-score">${candidate.score}</div>
          </div>
          <div class="metrics-inline">
            <span class="metric-chip">Goals ${candidate.goals}</span>
            <span class="metric-chip">Assists ${candidate.assists}</span>
            <span class="metric-chip">G+A ${candidate.goalContrib}</span>
            <span class="metric-chip">Pred goals ${candidate.predictedGoals}</span>
            <span class="metric-chip">Delta ${candidate.goalDelta}</span>
            <span class="metric-chip">Role conf ${candidate.positionConfidence}%</span>
          </div>
        </article>
      `
    )
    .join("");

  document.getElementById("award-leaders").innerHTML = awardRadar.leagueLeaders
    .map(
      (leader) => `
        <div class="leader-card">
          <span>${leader.league}</span>
          <strong>${leader.player}</strong>
          <small>${leader.squad}</small>
          <b>${leader.score}</b>
        </div>
      `
    )
    .join("");
}

function renderClassBreakdown(breakdown) {
  document.getElementById("combined-classes").innerHTML = renderClassCards(breakdown.combined);
  document.getElementById("detailed-classes").innerHTML = renderClassCards(breakdown.detailed);
}

function renderClassCards(cards) {
  return cards
    .map(
      (card) => `
        <div class="class-card">
          <div class="class-card-top">
            <div>
              <div class="class-card-name">${card.label}</div>
              <div>${card.fullLabel}</div>
            </div>
            <div class="note-chip">${card.support} rows</div>
          </div>
          <div class="class-stats">
            <span class="metric-chip">Precision ${card.precision}%</span>
            <span class="metric-chip">Recall ${card.recall}%</span>
            <span class="metric-chip">F1 ${card.f1}%</span>
          </div>
        </div>
      `
    )
    .join("");
}

function renderMatrices(matrices) {
  document.getElementById("combined-matrix").innerHTML = renderMatrix(matrices.combined);
  document.getElementById("detailed-matrix").innerHTML = renderMatrix(matrices.detailed);
}

function renderMatrix(matrixData) {
  const maxValue = Math.max(...matrixData.matrix.flat(), 1);
  const header = `
    <div class="matrix-header">
      <div></div>
      ${matrixData.labels.map((label) => `<div>${label}</div>`).join("")}
    </div>
  `;

  const rows = matrixData.matrix
    .map((row, rowIndex) => {
      const cells = row
        .map((value) => {
          const opacity = 0.12 + (value / maxValue) * 0.88;
          return `
            <div class="matrix-cell" style="background: rgba(159, 232, 112, ${opacity})">
              ${value}
            </div>
          `;
        })
        .join("");
      return `
        <div class="matrix-row">
          <div class="matrix-row-label">${matrixData.labels[rowIndex]}</div>
          ${cells}
        </div>
      `;
    })
    .join("");

  return `<div class="matrix-card">${header}${rows}</div>`;
}

function renderFeatureHighlights(features) {
  document.getElementById("classification-features").innerHTML = renderFeatureList(features.classification);
  document.getElementById("detailed-classification-features").innerHTML = renderFeatureList(features.detailedClassification);
  document.getElementById("regression-features").innerHTML = renderFeatureList(features.regression, true);
}

function renderFeatureList(rows, showDirection = false) {
  return `
    <div class="feature-list">
      ${rows
        .map(
          (row) => `
            <div class="feature-row">
              <strong>${row.label}</strong>
              <span class="feature-value ${showDirection ? row.direction : ""}">
                ${row.value}
              </span>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderPredictor(predictor) {
  document.getElementById("predictor-note").textContent = predictor.note;

  const positionForm = document.getElementById("position-form");
  const goalsForm = document.getElementById("goals-form");

  buildModelForm(positionForm, POSITION_FORM_GROUPS, predictor.schema, "classification");
  buildModelForm(goalsForm, GOALS_FORM_GROUPS, predictor.schema, "goals");

  positionForm.addEventListener("submit", handlePositionSubmit);
  goalsForm.addEventListener("submit", handleGoalsSubmit);

  updateModelDescriptions();

  const positionSelect = document.getElementById("position-model");
  const goalsSelect = document.getElementById("goals-model");
  positionSelect.addEventListener("change", updateModelDescriptions);
  goalsSelect.addEventListener("change", updateModelDescriptions);
}

function renderDatasetLab(schema) {
  const form = document.getElementById("dataset-form");
  const positionOptions = schema.models.position;
  const goalsOptions = schema.models.goals;

  form.innerHTML = `
    <section class="form-group">
      <h4>Dataset Upload</h4>
      <div class="field file-field">
        <label for="dataset-file">Choose CSV file</label>
        <input id="dataset-file" name="dataset-file" type="file" accept=".csv,text/csv" required />
      </div>
    </section>

    <section class="form-group">
      <h4>Models In Use</h4>
      <div class="field-grid">
        <div class="field">
          <label for="dataset-position-model">Position model</label>
          <select id="dataset-position-model" name="positionModel">
            ${positionOptions
              .map(
                (option) => `
                  <option value="${option.id}" ${option.id === schema.models.defaultPositionModel ? "selected" : ""}>
                    ${option.label}
                  </option>
                `
              )
              .join("")}
          </select>
        </div>
        <div class="field">
          <label for="dataset-goals-model">Goals model</label>
          <select id="dataset-goals-model" name="goalsModel">
            ${goalsOptions
              .map(
                (option) => `
                  <option value="${option.id}" ${option.id === schema.models.defaultGoalsModel ? "selected" : ""}>
                    ${option.label}
                  </option>
                `
              )
              .join("")}
          </select>
        </div>
      </div>
      <label class="sync-toggle">
        <input id="dataset-sync-position" type="checkbox" checked />
        Use the predicted position for the goals model on every row
      </label>
    </section>

    <p class="form-note">
      Recommended columns: Player, Nation, Squad, Comp, Pos, season, Age, Born, MP, Starts, Min, 90s,
      Gls, Ast, PK, PKatt, CrdY, CrdR, xG, npxG, xAG, PrgC, PrgP, PrgR. Missing fields will fall back
      to the training defaults.
    </p>
    <button class="submit-button" type="submit">Run Dataset Predictions</button>
  `;

  attachDatasetFormHandler();
}

function buildModelForm(form, groups, schema, mode) {
  const prefix = mode === "classification" ? "position" : "goals";
  const modelKey = mode === "classification" ? "position" : "goals";
  const modelOptions = schema.models[modelKey];
  const defaultModel = mode === "classification" ? schema.models.defaultPositionModel : schema.models.defaultGoalsModel;
  const formBody = groups
    .map(
      (group) => `
        <section class="form-group">
          <h4>${group.title}</h4>
          <div class="field-grid">
            ${group.fields.map((field) => renderField(field, schema, prefix)).join("")}
          </div>
        </section>
      `
    )
    .join("");

  const note = mode === "classification" ? schema.inputNotes.classification : schema.inputNotes.goals;
  const buttonLabel = mode === "classification" ? "Predict Position" : "Predict Goals";

  form.innerHTML = `
    <section class="form-group">
      <h4>Model In Use</h4>
      <div class="field">
        <label for="${prefix}-model">Choose model</label>
        <select name="model" id="${prefix}-model">
          ${modelOptions
            .map(
              (option) => `
                <option value="${option.id}" ${option.id === defaultModel ? "selected" : ""}>
                  ${option.label}
                </option>
              `
            )
            .join("")}
        </select>
      </div>
      <p id="${prefix}-model-description" class="form-note"></p>
    </section>
    ${formBody}
    <p class="form-note">${note}</p>
    <button class="submit-button" type="submit">${buttonLabel}</button>
  `;
}

function renderField(field, schema, prefix) {
  const label = FIELD_LABELS[field] || field;
  const options = schema.options[field];
  const defaultValue = schema.defaults[field] ?? "";
  const inputType = typeof defaultValue === "number" ? "number" : "text";
  const inputId = `${prefix}-${field}`;

  if (field === "season" || field === "Pos") {
    const selectOptions = options
      .map(
        (option) => `
          <option value="${option}" ${String(defaultValue) === String(option) ? "selected" : ""}>
            ${option}
          </option>
        `
      )
      .join("");
    return `
      <div class="field">
        <label for="${inputId}">${label}</label>
        <select name="${field}" id="${inputId}">
          ${selectOptions}
        </select>
      </div>
    `;
  }

  if (Array.isArray(options)) {
    const datalistId = `${prefix}-${field}-options`;
    return `
      <div class="field">
        <label for="${inputId}">${label}</label>
        <input
          id="${inputId}"
          name="${field}"
          type="text"
          list="${datalistId}"
          value="${defaultValue}"
          autocomplete="off"
        />
        <datalist id="${datalistId}">
          ${options.map((option) => `<option value="${option}"></option>`).join("")}
        </datalist>
      </div>
    `;
  }

  return `
    <div class="field">
      <label for="${inputId}">${label}</label>
      <input
        id="${inputId}"
        name="${field}"
        type="${inputType}"
        inputmode="decimal"
        step="any"
        value="${defaultValue}"
      />
    </div>
  `;
}

async function handlePositionSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button[type='submit']");
  setButtonBusy(button, true, "Running...");

  try {
    const payload = formToObject(form);
    const result = await fetchJSON("/api/predict/position", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.lastPositionPrediction = result.prediction;
    if (document.getElementById("sync-position-toggle").checked) {
      const goalsPos = document.querySelector("#goals-form [name='Pos']");
      if (goalsPos) {
        goalsPos.value = result.prediction;
      }
    }
    renderPositionResult(result);
  } catch (error) {
    renderErrorPanel("position-result", error.message);
  } finally {
    setButtonBusy(button, false, "Predict Position");
  }
}

async function handleGoalsSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button[type='submit']");
  setButtonBusy(button, true, "Running...");

  try {
    const payload = formToObject(form);
    if (document.getElementById("sync-position-toggle").checked && state.lastPositionPrediction) {
      payload.Pos = state.lastPositionPrediction;
      const goalsPos = form.querySelector("[name='Pos']");
      if (goalsPos) {
        goalsPos.value = state.lastPositionPrediction;
      }
    }

    const result = await fetchJSON("/api/predict/goals", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderGoalsResult(result);
  } catch (error) {
    renderErrorPanel("goals-result", error.message);
  } finally {
    setButtonBusy(button, false, "Predict Goals");
  }
}

async function handleDatasetSubmit(event) {
  event.preventDefault();
  console.log("Dataset form submitted");
  const form = event.currentTarget;
  const button = form.querySelector("button[type='submit']");
  const fileInput = document.getElementById("dataset-file");
  const file = fileInput?.files?.[0];

  if (!file) {
    console.warn("No file selected");
    renderDatasetError("Choose a CSV file first.");
    return;
  }

  console.log("Processing file:", file.name);
  setButtonBusy(button, true, "Running...");

  try {
    const csvText = await file.text();
    console.log("CSV loaded, sending to /api/predict/dataset");
    const payload = {
      filename: file.name,
      csvText,
      positionModel: document.getElementById("dataset-position-model").value,
      goalsModel: document.getElementById("dataset-goals-model").value,
      usePredictedPosition: document.getElementById("dataset-sync-position").checked,
    };

    const result = await fetchJSON("/api/predict/dataset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    state.lastDatasetResult = result;
    renderDatasetBatchResult(result);
  } catch (error) {
    console.error("Dataset submission error:", error);
    renderDatasetError(error.message);
  } finally {
    setButtonBusy(button, false, "Run Dataset Predictions");
  }
}

function renderPositionResult(result) {
  document.getElementById("position-result").classList.remove("empty-state");
  document.getElementById("position-result").innerHTML = `
    <div class="card-kicker">Predicted Position</div>
    <p class="result-headline">${result.predictionLabel}</p>
    <div class="note-chip">Model: ${result.model}</div>
    <p class="result-copy">${result.modelDescription}</p>
    <div class="metrics-inline">
      <span class="metric-chip">Top probability ${result.topProbabilityPct}%</span>
      <span class="metric-chip">Code ${result.prediction}</span>
    </div>
    <div class="probability-stack">
      ${result.probabilities
        .map(
          (item, index) => `
            <div class="probability-row">
              <div class="probability-label">${item.label}</div>
              <div class="bar-track">
                <div class="bar-fill ${index === 0 ? "" : "muted"}" style="width: ${item.probabilityPct}%"></div>
              </div>
              <div class="bar-value">${item.probabilityPct}%</div>
            </div>
          `
        )
        .join("")}
    </div>
    <div class="derived-grid">
      ${renderDerivedItem("G+A", result.derived.goalsPlusAssists)}
      ${renderDerivedItem("G-PK", result.derived.nonPenaltyGoals)}
      ${renderDerivedItem("npxG+xAG", result.derived.npxGPlusxAG)}
      ${renderDerivedItem("Goals /90", result.derived.goalsPer90)}
      ${renderDerivedItem("xG /90", result.derived.xGPer90)}
      ${renderDerivedItem("Start rate", result.derived.startRate)}
    </div>
    ${renderWarnings(result.warnings)}
  `;
}

function renderGoalsResult(result) {
  document.getElementById("goals-result").classList.remove("empty-state");
  document.getElementById("goals-result").innerHTML = `
    <div class="card-kicker">Predicted Goals</div>
    <p class="result-headline">${result.predictedGoals}</p>
    <div class="note-chip">Model: ${result.model}</div>
    <p class="result-copy">${result.modelDescription}</p>
    <div class="metrics-inline">
      <span class="metric-chip">Training percentile ${result.percentile}%</span>
      <span class="metric-chip">Raw model output ${result.rawPrediction}</span>
    </div>
    <div class="derived-grid">
      ${renderDerivedItem("xG /90", result.derived.xGPer90)}
      ${renderDerivedItem("xAG /90", result.derived.xAGPer90)}
      ${renderDerivedItem("Prog passes /90", result.derived.progressivePassesPer90)}
      ${renderDerivedItem("Start rate", result.derived.startRate)}
    </div>
    ${renderWarnings(result.warnings)}
  `;
}

function renderDatasetBatchResult(result) {
  const summary = document.getElementById("dataset-summary");
  summary.classList.remove("empty-state");
  summary.innerHTML = `
    <div class="card-kicker">${result.filename}</div>
    <p class="result-headline">${result.rowsProcessed}</p>
    <p class="result-copy">
      Processed rows with ${result.positionModel} for classification and ${result.goalsModel} for regression.
    </p>
    <div class="metrics-inline">
      <span class="metric-chip">Avg predicted goals ${result.summary.averagePredictedGoals}</span>
      <span class="metric-chip">Peak predicted goals ${result.summary.maxPredictedGoals}</span>
      <span class="metric-chip">Mean role confidence ${result.summary.meanPositionConfidence}%</span>
      <span class="metric-chip">${result.usePredictedPosition ? "Goals model uses predicted positions" : "Goals model uses dataset positions"}</span>
    </div>
    <div class="dataset-breakdown">
      ${result.summary.positionBreakdown
        .map((item) => `<span class="note-chip">${item.label} ${item.count}</span>`)
        .join("")}
    </div>
    <div class="metrics-inline">
      <span class="metric-chip">Detected columns ${result.columnsDetected}</span>
      <span class="metric-chip">Defaulted columns ${result.filledColumns.length}</span>
      <span class="metric-chip">Low-minute rows ${result.lowMinuteRows}</span>
    </div>
    ${renderWarnings(result.warnings)}
  `;

  const top10 = document.getElementById("dataset-top10");
  top10.innerHTML = result.ballonCandidates
    .map(
      (player, index) => `
        <article class="dataset-player-card ${index === 0 ? "featured" : ""}">
          <div class="dataset-player-head">
            <div>
              <h4>${player.player}</h4>
              <p>${player.squad} / ${player.league} / ${player.predictedPositionLabel}</p>
            </div>
            <div class="dataset-rank">#${player.rank}</div>
          </div>
          <div class="metrics-inline">
            <span class="metric-chip">Ballon score ${player.ballonScore}</span>
            <span class="metric-chip">Pred goals ${player.predictedGoals}</span>
            <span class="metric-chip">Pos conf ${player.positionConfidencePct}%</span>
            <span class="metric-chip">Actual goals ${player.actualGoals}</span>
          </div>
        </article>
      `
    )
    .join("");

  renderLeagueTop5(result.topLeaguePlayers);

}

function renderDatasetError(message) {
  renderErrorPanel("dataset-summary", message);
  document.getElementById("dataset-top10").innerHTML = `
    <div class="warning-stack">
      <div class="warning-item">${message}</div>
    </div>
  `;
  document.getElementById("dataset-league-top5").innerHTML = `
    <div class="warning-stack">
      <div class="warning-item">${message}</div>
    </div>
  `;
  document.getElementById("dataset-preview").classList.remove("empty-state");
  document.getElementById("dataset-preview").innerHTML = `
    <div class="warning-stack">
      <div class="warning-item">${message}</div>
    </div>
  `;
}

function renderDerivedItem(label, value) {
  return `
    <div class="derived-item">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `;
}

function renderWarnings(warnings) {
  if (!warnings || warnings.length === 0) {
    return "";
  }
  return `
    <div class="warning-stack">
      ${warnings.map((warning) => `<div class="warning-item">${warning}</div>`).join("")}
    </div>
  `;
}

function renderLeagueTop5(leagues) {
  const leagueGrid = document.getElementById("dataset-league-top5");
  leagueGrid.innerHTML = leagues
    .map(
      (league) => `
        <section class="league-card">
          <div class="league-title">${league.league}</div>
          <div class="league-player-list">
            ${league.players
              .map(
                (player) => `
                  <div class="league-player-row">
                    <strong>#${player.rank} ${player.player}</strong>
                    <span>${player.squad}</span>
                    <span>${player.predictedPositionLabel}</span>
                    <span>Goals ${player.predictedGoals}</span>
                    <span>Conf ${player.positionConfidencePct}%</span>
                  </div>
                `
              )
              .join("")}
          </div>
        </section>
      `
    )
    .join("");
}

function renderErrorPanel(targetId, message) {
  const target = document.getElementById(targetId);
  target.classList.remove("empty-state");
  target.innerHTML = `
    <div class="warning-stack">
      <div class="warning-item">${message}</div>
    </div>
  `;
}

function updateModelDescriptions() {
  const schema = state.dashboard.predictor.schema;
  const positionModelId = document.getElementById("position-model").value;
  const goalsModelId = document.getElementById("goals-model").value;
  const positionOption = schema.models.position.find((option) => option.id === positionModelId);
  const goalsOption = schema.models.goals.find((option) => option.id === goalsModelId);

  document.getElementById("position-model-description").textContent = `${positionOption.dataset}: ${positionOption.description}`;
  document.getElementById("goals-model-description").textContent = `${goalsOption.dataset}: ${goalsOption.description}`;
}

function formToObject(form) {
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}

function setButtonBusy(button, busy, label) {
  button.disabled = busy;
  button.textContent = label;
}

function groupBy(items, key) {
  return items.reduce((accumulator, item) => {
    const value = item[key];
    if (!accumulator[value]) {
      accumulator[value] = [];
    }
    accumulator[value].push(item);
    return accumulator;
  }, {});
}
