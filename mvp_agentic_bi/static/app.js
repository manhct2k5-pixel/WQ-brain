const state = {
  dataset: null,
};

const el = {
  modeBadge: document.getElementById("modeBadge"),
  modelText: document.getElementById("modelText"),
  fileInput: document.getElementById("fileInput"),
  uploadButton: document.getElementById("uploadButton"),
  uploadStatus: document.getElementById("uploadStatus"),
  datasetPanel: document.getElementById("datasetPanel"),
  datasetMeta: document.getElementById("datasetMeta"),
  sheetSelect: document.getElementById("sheetSelect"),
  warningBox: document.getElementById("warningBox"),
  summaryCards: document.getElementById("summaryCards"),
  columnsTable: document.getElementById("columnsTable"),
  insightList: document.getElementById("insightList"),
  sampleTable: document.getElementById("sampleTable"),
  questionPanel: document.getElementById("questionPanel"),
  exampleQuestions: document.getElementById("exampleQuestions"),
  questionInput: document.getElementById("questionInput"),
  askButton: document.getElementById("askButton"),
  questionStatus: document.getElementById("questionStatus"),
  answerPanel: document.getElementById("answerPanel"),
  answerMode: document.getElementById("answerMode"),
  answerText: document.getElementById("answerText"),
  planBox: document.getElementById("planBox"),
  resultTable: document.getElementById("resultTable"),
};

function setModeBadge(mode, model) {
  if (mode === "openai") {
    el.modeBadge.textContent = "Planner AI: Bật";
    el.modelText.textContent = model ? `Model: ${model}` : "";
  } else {
    el.modeBadge.textContent = "Planner AI: Tắt";
    el.modelText.textContent = "App đang chạy ở chế độ heuristic cục bộ.";
  }
}

function renderCards(profile) {
  const cards = [
    ["Số dòng", profile.row_count],
    ["Số cột", profile.column_count],
    ["Cột số", profile.numeric_columns.length],
    ["Cột phân loại", profile.categorical_columns.length],
    ["Cột ngày", profile.date_columns.length],
  ];
  el.summaryCards.innerHTML = cards
    .map(
      ([label, value]) => `
        <div class="card">
          <span class="label">${label}</span>
          <div class="value">${value}</div>
        </div>
      `
    )
    .join("");
}

function renderTable(columns, rows) {
  if (!rows || rows.length === 0) {
    return "<p class='muted'>Không có dữ liệu để hiển thị.</p>";
  }
  const head = columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = columns.map((col) => `<td>${escapeHtml(String(row[col] ?? ""))}</td>`).join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  return `
    <div class="table-wrap">
      <table>
        <thead><tr>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
  `;
}

function renderDataset(data) {
  if (!data.has_dataset) {
    el.datasetPanel.classList.add("hidden");
    el.questionPanel.classList.add("hidden");
    return;
  }
  state.dataset = data;
  setModeBadge(data.mode, data.model);
  el.datasetPanel.classList.remove("hidden");
  el.questionPanel.classList.remove("hidden");

  el.datasetMeta.textContent = `File: ${data.filename} • Sheet: ${data.active_sheet} • Tải lúc: ${data.uploaded_at}`;
  renderCards(data.profile);

  el.sheetSelect.innerHTML = data.sheet_names
    .map((name) => `<option value="${escapeHtml(name)}" ${name === data.active_sheet ? "selected" : ""}>${escapeHtml(name)}</option>`)
    .join("");

  if (data.warnings && data.warnings.length) {
    el.warningBox.classList.remove("hidden");
    el.warningBox.innerHTML = data.warnings.map((item) => `<div>${escapeHtml(item)}</div>`).join("");
  } else {
    el.warningBox.classList.add("hidden");
    el.warningBox.innerHTML = "";
  }

  const columnRows = data.profile.columns.map((column) => ({
    name: column.name,
    type: column.type,
    null_ratio: `${column.null_ratio}%`,
    unique_count: column.unique_count,
    sample_values: (column.sample_values || []).join(", "),
  }));
  el.columnsTable.innerHTML = renderTable(
    ["name", "type", "null_ratio", "unique_count", "sample_values"],
    columnRows
  );

  el.insightList.innerHTML = (data.profile.insights || [])
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");

  el.sampleTable.innerHTML = renderTable(
    data.profile.columns.map((item) => item.name),
    data.profile.sample_rows || []
  );

  el.exampleQuestions.innerHTML = (data.profile.example_questions || [])
    .map(
      (question) => `<button class="secondary example-chip" data-question="${escapeAttribute(question)}">${escapeHtml(question)}</button>`
    )
    .join("");
}

async function loadState() {
  const response = await fetch("/api/state");
  const data = await response.json();
  setModeBadge(data.mode, data.model);
  renderDataset(data);
}

async function uploadFile() {
  const file = el.fileInput.files[0];
  if (!file) {
    el.uploadStatus.textContent = "Hãy chọn file Excel/CSV trước.";
    return;
  }
  el.uploadStatus.textContent = "Đang tải và phân tích dữ liệu...";
  const base64 = await fileToBase64(file);
  const response = await fetch("/api/upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: file.name,
      content_base64: base64,
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    el.uploadStatus.textContent = data.error || "Không tải được file.";
    return;
  }
  el.uploadStatus.textContent = "Tải file thành công.";
  renderDataset(data.state);
}

async function selectSheet(sheetName) {
  const response = await fetch("/api/select-sheet", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sheet_name: sheetName }),
  });
  const data = await response.json();
  if (!response.ok) {
    el.uploadStatus.textContent = data.error || "Không đổi được sheet.";
    return;
  }
  renderDataset(data.state);
}

async function askQuestion() {
  const question = el.questionInput.value.trim();
  if (!question) {
    el.questionStatus.textContent = "Hãy nhập câu hỏi trước khi phân tích.";
    return;
  }
  el.questionStatus.textContent = "Đang phân tích...";
  const response = await fetch("/api/question", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await response.json();
  if (!response.ok) {
    el.questionStatus.textContent = data.error || "Không phân tích được câu hỏi.";
    return;
  }
  el.questionStatus.textContent = "Phân tích xong.";
  el.answerPanel.classList.remove("hidden");
  el.answerMode.textContent = data.mode === "openai" ? "Planner: OpenAI" : "Planner: heuristic cục bộ";
  el.answerText.textContent = data.answer || "";
  el.planBox.textContent = JSON.stringify(data.plan || {}, null, 2);
  if (data.result_columns && data.result_columns.length) {
    el.resultTable.innerHTML = renderTable(data.result_columns, data.result_rows || []);
  } else {
    el.resultTable.innerHTML = "<p class='muted'>Không có bảng kết quả chi tiết.</p>";
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      const encoded = result.includes(",") ? result.split(",")[1] : result;
      resolve(encoded);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttribute(value) {
  return escapeHtml(value);
}

el.uploadButton.addEventListener("click", uploadFile);
el.askButton.addEventListener("click", askQuestion);
el.sheetSelect.addEventListener("change", (event) => selectSheet(event.target.value));
el.exampleQuestions.addEventListener("click", (event) => {
  const button = event.target.closest("[data-question]");
  if (!button) return;
  el.questionInput.value = button.dataset.question;
  askQuestion();
});

loadState();
