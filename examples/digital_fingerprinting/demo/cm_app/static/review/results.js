/*
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

let columnIndexes = [];

// Function to read and display CSV file contents
function handleFileSelect() {
    const select = document.getElementById('columns');
    select.innerHTML = '';
    const file = document.getElementById('csv-file').files[0];
    const reader = new FileReader();
    reader.readAsText(file);
    reader.onload = function(event) {
        const csv = event.target.result;
        const rows = csv.split('\n');
        const table = document.createElement('table');
        table.border = '1';
        const headerRow = document.createElement('tr');
        const headerCells = parseCSVRow(rows[0]);

        // Create a new header cell for the checkboxes
        const markHeaderCell = document.createElement('th');
        markHeaderCell.innerHTML = 'Mark';
        headerRow.appendChild(markHeaderCell);

        for (let j = 0; j < headerCells.length; j++) {
            const headerCell = document.createElement('th');
            headerCell.innerHTML = headerCells[j];
            headerRow.appendChild(headerCell);
            const option = document.createElement('option');
            option.value = j;
            option.text = headerCells[j];
            document.getElementById('columns').add(option);
        }
        table.appendChild(headerRow);
        for (let i = 1; i < rows.length; i++) {
            const cells = parseCSVRow(rows[i]);
            // skip rows with no data
            if (cells.length === 1 && cells[0] === '') {
                continue;
            }
            const row = document.createElement('tr');

            // Add checkbox as a cell in the first column
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = i;
            checkbox.onclick = toggleMark;
            const markCell = document.createElement('td');
            markCell.appendChild(checkbox);
            row.appendChild(markCell);

            for (let j = 0; j < cells.length; j++) {
                const cell = document.createElement('td');
                cell.innerHTML = cells[j];
                row.appendChild(cell);
            }
            table.appendChild(row);
        }
        const csvContent = document.getElementById('csv-content');
        csvContent.innerHTML = ''; // Clear existing content
        csvContent.appendChild(table);
    };
}

// Function to parse a row of CSV data, accounting for commas within double quotes
function parseCSVRow(row) {
    const cells = [];
    let inQuotes = false;
    let start = 0;
    for (let i = 0; i < row.length; i++) {
        if (row[i] === '"') {
            inQuotes = !inQuotes;
        } else if (row[i] === ',' && !inQuotes) {
            cells.push(row.substring(start, i));
            start = i + 1;
        }
    }
    cells.push(row.substring(start));
    return cells;
}

// Function to toggle column visibility
function toggleColumn() {
    const select = document.getElementById('columns');
    columnIndexes = [];
    for (let i = 0; i < select.options.length; i++) {
        if (select.options[i].selected) {
            columnIndexes.push(i);
        }
    }
    const table = document.querySelector('table');
    const cells = table.querySelectorAll('td,th');
    for (let i = 0; i < cells.length; i++) {
        const columnIndex = getColumnIndex(cells[i]);
        if (columnIndexes.includes(columnIndex)) {
            cells[i].classList.remove('hidden');
        } else {
            cells[i].classList.add('hidden');
        }
    }
}
// Function to get the index of the column containing a cell


// Function to toggle row mark
function toggleMark() {
    const row = this.parentNode.parentNode;
    if (this.checked) {
        row.classList.add('marked');
    } else {
        row.classList.remove('marked');
    }
}

function getColumnIndex(cell) {
    const row = cell.parentNode;
    const headerRow = row.parentNode.querySelector('tr:first-child');
    const cells = row.querySelectorAll('td,th');
    for (let i = 0; i < cells.length; i++) {
        if (cells[i] === cell) {
            return i - 1;
        }
    }
    return -1;
}

function saveMarked() {
    // Check if any rows have been marked
    if ($('#csv-content tr.marked').length === 0) {
        alert('Please select at least one row before saving.');
        return;
    }
    // Ask user for filename
    const filename = prompt("Enter filename:");

    const table = document.querySelector('table');
    const rows = table.querySelectorAll('tr:not(:first-child)');
    const selectedRows = [];
    for (let i = 0; i < rows.length; i++) {
        if (rows[i].classList.contains('marked')) {
            const cells = rows[i].querySelectorAll('td:not(:first-child),th:not(:first-child)');
            const selectedCells = [];
            for (let j = 0; j < cells.length; j++) {
                const columnIndex = getColumnIndex(cells[j]);
                if (columnIndexes.includes(columnIndex)) {
                    selectedCells.push(cells[j].innerHTML);
                }
            }
            selectedRows.push(selectedCells.join(','));
        }
    }

    // Add header to CSV
    const headerCells = [];
    const select = document.getElementById('columns');
    for (let i = 0; i < select.options.length; i++) {
        if (select.options[i].selected) {
            headerCells.push(select.options[i].text);
        }
    }
    const header = headerCells.join(',');
    const csvContent = 'data:text/csv;charset=utf-8,' + header + '\n' + selectedRows.join('\n');

    // Create a link and click it to download the CSV file
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `${filename}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Display message to navigate to download location and copy file location manually
    alert(`The file "${filename}.csv" has been downloaded. Please navigate to the download location and copy the file location manually.`);

    $('#submit-button').prop('disabled', false);
  }

  function submitTrainingMessage() {
    // Make HTTP GET request
    $.ajax({
        url: "/training",
        method: "GET",
        success: function(response) {
            window.location.href = "/training";
        }
    });
  }
