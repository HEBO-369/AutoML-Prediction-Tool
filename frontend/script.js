document.addEventListener('DOMContentLoaded', () => {

    // elements
    const uploadForm = document.getElementById('upload-form');
    const csvFileInput = document.getElementById('csv-file');
    const fileNameDisplay = document.getElementById('file-name');
    
    const step2 = document.getElementById('step2');
    const targetSelect = document.getElementById('target-column-select');
    const trainButton = document.getElementById('train-button');
    const loadingSpinner = document.getElementById('loading-spinner');

    const step3 = document.getElementById('step3');
    const modelResultsDiv = document.getElementById('model-results');

    const step4 = document.getElementById('step4');
    const predictionForm = document.getElementById('prediction-form');
    const predictButton = document.getElementById('predict-button');
    const predictionSpinner = document.getElementById('prediction-spinner');
    const predictionResultsDiv = document.getElementById('prediction-results');

    // backend Api
    const API_URL = "http://127.0.0.1:8000";

    // upload file
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const file = csvFileInput.files[0];
        if (!file) return;

        fileNameDisplay.textContent = `sending ${file.name}...`;
        
        const formData = new FormData();
        formData.append("file", file);

        try {
            // send to backend
            const response = await fetch(`${API_URL}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'upload error');
            }

            const result = await response.json();
            
            //show the list of columns
            targetSelect.innerHTML = '<option value="">select columns...</option>';
            result.columns.forEach(header => {
                const option = document.createElement('option');
                option.value = header;
                option.textContent = header;
                targetSelect.appendChild(option);
            });
            
            fileNameDisplay.textContent = `file uploaded : ${file.name}`;
            step2.style.display = 'block';

        } catch (error) {
            fileNameDisplay.textContent = `error : ${error.message}`;
            alert(`error : ${error.message}`);
        }
    });

    //train model
    trainButton.addEventListener('click', async () => {
        const targetColumn = targetSelect.value;
        if (!targetColumn) {
            alert('select target column first');
            return;
        }

        loadingSpinner.style.display = 'block';
        step3.style.display = 'none';
        step4.style.display = 'none';
        modelResultsDiv.innerHTML = '';
        predictionForm.innerHTML = '';

        try {
            const trainData = new FormData();
            trainData.append('target_column', targetColumn);

            //send to backend
            const response = await fetch(`${API_URL}/train`, {
                method: 'POST',
                body: trainData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'training error');
            }

            const result = await response.json();

            // show model results
            let statsHtml = '';
            if (result.dropped_columns && result.dropped_columns.length > 0) {
                statsHtml += `
                    <p style="color: #990000; font-weight: bold; font-size: 0.9em;">
                        dropped columns due to high unwanted values: 
                        ${result.dropped_columns.join(', ')}
                    </p>
                `;
            }
            if (result.outliers_removed && result.outliers_removed > 0) {
                statsHtml += `
                    <p style="color: #E67E22; font-weight: bold; font-size: 0.9em;">
                        ${result.outliers_removed} outliers were removed from the dataset.
                    </p>
                `;
            }

            // show accuracies table
            let resultsTable = '<table><tr><th> The Model</th><th>Accuracy(Accuracy)</th></tr>';
            for (const [model, accuracy] of Object.entries(result.accuracies)) {
                resultsTable += `<tr><td>${model}</td><td>${accuracy}</td></tr>`;
            }
            resultsTable += '</table>';
            
            // adding to the DOM
            modelResultsDiv.innerHTML = statsHtml + resultsTable;
            step3.style.display = 'block';

            // add the prediction form
            predictionForm.innerHTML = ''; //clear previous form
            const features = result.prediction_features;
            
            for (const [col, details] of Object.entries(features)) {
                const formGroup = document.createElement('div');
                formGroup.className = 'form-group';
                
                const label = document.createElement('label');
                label.setAttribute('for', `pred-${col}`);
                label.textContent = col;
                formGroup.appendChild(label);

                if (details.type === 'number') {
                    const input = document.createElement('input');
                    input.type = 'number';
                    input.id = `pred-${col}`;
                    input.name = col;
                    input.placeholder = `insert ${col}...`;
                    input.step = "any"; 
                    formGroup.appendChild(input);
                } else if (details.type === 'dropdown') {
                    const select = document.createElement('select');
                    select.id = `pred-${col}`;
                    select.name = col;
                    
                    //add default option
                    const defaultOption = document.createElement('option');
                    defaultOption.value = "";
                    defaultOption.textContent = `choose ${col}...`;
                    select.appendChild(defaultOption);
                    
                    details.values.forEach(val => {
                        const option = document.createElement('option');
                        option.value = val;
                        option.textContent = val;
                        select.appendChild(option);
                    });
                    formGroup.appendChild(select);
                }
                
                predictionForm.appendChild(formGroup);
            }
            
            step4.style.display = 'block';

        } catch (error) {
            alert(`error: ${error.message}`);
        } finally {
            loadingSpinner.style.display = 'none';
        }
    });

    // predict
    predictButton.addEventListener('click', async () => {
        predictionSpinner.style.display = 'block';
        predictionResultsDiv.innerHTML = '';

        // collect the data
        const formData = new FormData(predictionForm);
        const features = {};
        formData.forEach((value, key) => {
            features[key] = value;
        });

        try {
            // send to backend
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ features: features })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'prediction error');
            }

            const result = await response.json();

            // show prediction results
            let resultsTable = '<table><tr><th>The Model</th><th>Prediction</th></tr>';
            for (const [model, prediction] of Object.entries(result.predictions)) {
                resultsTable += `<tr><td>${model}</td><td>${prediction}</td></tr>`;
            }
            resultsTable += '</table>';
            
            predictionResultsDiv.innerHTML = resultsTable;

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            predictionSpinner.style.display = 'none';
        }
    });
});