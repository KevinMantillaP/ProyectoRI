function search() {
    const query = document.getElementById('query').value;
    
    if (!query) {
        alert('Por favor ingrese una consulta');
        return;
    }

    // Aquí deberías realizar la llamada al backend para procesar la query.
    // Supongamos que hacemos una llamada fetch a una API del backend.
    fetch(`/search?q=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function displayResults(results) {
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = ''; // Limpiamos los resultados anteriores

    if (results.length === 0) {
        resultsContainer.innerHTML = '<p>No se encontraron resultados</p>';
        return;
    }

    results.forEach(result => {
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        resultItem.textContent = result; // Aquí puedes personalizar cómo mostrar cada resultado
        resultsContainer.appendChild(resultItem);
    });
}
