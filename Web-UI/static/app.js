
console.log("Smart Entry UI loaded successfully.");

const searchInput = document.getElementById("logSearch");
const dateInput = document.getElementById("dateSearch");

function filterLogs() {
    const searchText = searchInput
        ? searchInput.value.trim().toLowerCase()
        : "";

    const selectedDate = dateInput ? dateInput.value : "";

    const rows = document.querySelectorAll(
        ".camera table tbody tr"
    );

    rows.forEach(function (row) {
        const cells = row.querySelectorAll("td");

        if (cells.length < 4) return;

        const name = cells[0].textContent.trim().toLowerCase();
        const date = cells[1].textContent.trim();

        const matchesName = name.includes(searchText);
        const matchesDate = !selectedDate || date === selectedDate;

        row.style.display =
            matchesName && matchesDate ? "" : "none";
    });
}

if (searchInput) {
    searchInput.addEventListener("input", filterLogs);
}

if (dateInput) {
    dateInput.addEventListener("change", filterLogs);
}
