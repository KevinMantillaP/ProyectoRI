document.getElementById('search-form').addEventListener('submit', function(event) {
    event.preventDefault();
    var query = document.getElementById('query').value;
    fetch('/search?q=' + encodeURIComponent(query))
        .then(response => response.json())
        .then(data => {
            var resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<h2>Results:</h2>';
            for (var key in data) {
                resultsDiv.innerHTML += '<h3>' + key + '</h3><ul>';
                data[key].forEach(function(item) {
                    resultsDiv.innerHTML += '<li>' + item + '</li>';
                });
                resultsDiv.innerHTML += '</ul>';
            }
        })
        .catch(error => console.error('Error:', error));
});


function displayResults(results) {
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = '';

    // Verificar y mostrar resultados de BoW
    if (results.BoW && Array.isArray(results.BoW)) {
        const bowTitle = document.createElement('h3');
        bowTitle.textContent = 'Resultados BoW';
        resultsContainer.appendChild(bowTitle);
        
        results.BoW.forEach(result => {
            const p = document.createElement('p');
            p.textContent = result;
            resultsContainer.appendChild(p);
        });
    }

    // Verificar y mostrar resultados de TF-IDF
    if (results['TF-IDF'] && Array.isArray(results['TF-IDF'])) {
        const tfidfTitle = document.createElement('h3');
        tfidfTitle.textContent = 'Resultados TF-IDF';
        resultsContainer.appendChild(tfidfTitle);

        results['TF-IDF'].forEach(result => {
            const p = document.createElement('p');
            p.textContent = result;
            resultsContainer.appendChild(p);
        });
    }
}
