const searchInput = document.getElementById('searchInput');
const searchResults = document.getElementById('searchResults');
const homeView = document.getElementById('homeView');
const detailView = document.getElementById('detailView');
const homeGrid = document.getElementById('homeGrid');
const recoGrid = document.getElementById('recoGrid');
const genreFilter = document.getElementById('genreFilter');
const gridTitle = document.getElementById('gridTitle');
const backBtn = document.getElementById('backBtn');

document.querySelectorAll('.notice').forEach((el) => el.remove());

let searchDebounce = null;

// ---------------- helpers ----------------
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${url}`);
  return res.json();
}

function movieCardHTML(movie) {
  const matchBadge = movie.match_score != null
    ? `<span class="match-badge">${movie.match_score}% match</span>`
    : `<span>&#9733; ${movie.vote_average}</span>`;
  return `
    <div class="movie-card" data-movie-id="${movie.movie_id}">
      <img src="${movie.poster_url}" alt="${escapeHTML(movie.title)} poster" loading="lazy" />
      <div class="movie-card-body">
        <p class="movie-card-title">${escapeHTML(movie.title)}</p>
        <div class="movie-card-meta">
          <span>${movie.release_year || ''}</span>
          ${matchBadge}
        </div>
      </div>
    </div>`;
}

function escapeHTML(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}

function attachCardHandlers(container) {
  container.querySelectorAll('.movie-card').forEach((card) => {
    card.addEventListener('click', () => openMovie(card.dataset.movieId));
  });
}

// ---------------- search ----------------
searchInput.addEventListener('input', () => {
  clearTimeout(searchDebounce);
  const q = searchInput.value.trim();
  if (q.length < 2) {
    searchResults.classList.add('hidden');
    return;
  }
  searchDebounce = setTimeout(async () => {
    const results = await fetchJSON(`/api/search?q=${encodeURIComponent(q)}`);
    if (!results.length) {
      searchResults.classList.add('hidden');
      return;
    }
    searchResults.innerHTML = results.map((m) => `
      <div class="search-result-item" data-movie-id="${m.movie_id}">
        <img src="${m.poster_url}" alt="" />
        <div>
          <div class="sri-title">${escapeHTML(m.title)}</div>
          <div class="sri-year">${m.release_year || ''}</div>
        </div>
      </div>`).join('');
    searchResults.classList.remove('hidden');
    searchResults.querySelectorAll('.search-result-item').forEach((item) => {
      item.addEventListener('click', () => {
        searchResults.classList.add('hidden');
        searchInput.value = '';
        openMovie(item.dataset.movieId);
      });
    });
  }, 220);
});

document.addEventListener('click', (e) => {
  if (!searchResults.contains(e.target) && e.target !== searchInput) {
    searchResults.classList.add('hidden');
  }
});

// ---------------- home grid / genres ----------------
async function loadTrending() {
  gridTitle.textContent = 'Highest rated on Movieverse';
  const movies = await fetchJSON('/api/trending?limit=18');
  homeGrid.innerHTML = movies.map(movieCardHTML).join('');
  attachCardHandlers(homeGrid);
}

async function loadGenre(genre) {
  gridTitle.textContent = `Top ${genre} films`;
  const movies = await fetchJSON(`/api/genre/${encodeURIComponent(genre)}?limit=18`);
  homeGrid.innerHTML = movies.map(movieCardHTML).join('');
  attachCardHandlers(homeGrid);
}

async function initGenreFilter() {
  const genres = await fetchJSON('/api/genres');
  const topGenres = genres.filter((g) => !['Foreign', 'TV Movie'].includes(g)).slice(0, 9);
  topGenres.forEach((g) => {
    const btn = document.createElement('button');
    btn.className = 'genre-chip';
    btn.dataset.genre = g;
    btn.textContent = g;
    genreFilter.appendChild(btn);
  });

  genreFilter.addEventListener('click', (e) => {
    const btn = e.target.closest('.genre-chip');
    if (!btn) return;
    genreFilter.querySelectorAll('.genre-chip').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    if (btn.dataset.genre === '__trending__') {
      loadTrending();
    } else {
      loadGenre(btn.dataset.genre);
    }
  });
}

// ---------------- detail view ----------------
async function openMovie(movieId) {
  const [movie, recs] = await Promise.all([
    fetchJSON(`/api/movie/${movieId}`),
    fetchJSON(`/api/recommend/${movieId}?limit=10`),
  ]);

  document.getElementById('detailPoster').src = movie.poster_url;
  document.getElementById('detailPoster').alt = `${movie.title} poster`;
  document.getElementById('detailTitle').textContent = movie.title;
  document.getElementById('detailYear').textContent = movie.release_year || '—';
  document.getElementById('detailRuntime').textContent = movie.runtime ? `${movie.runtime} min` : '';
  document.getElementById('detailRating').textContent = `★ ${movie.vote_average}`;
  document.getElementById('detailOverview').textContent = movie.overview || 'No synopsis available.';
  document.getElementById('detailDirector').textContent = movie.director || 'Unknown';
  document.getElementById('detailGenres').innerHTML = movie.genres
    .map((g) => `<span>${escapeHTML(g)}</span>`).join('');

  recoGrid.innerHTML = recs.map(movieCardHTML).join('');
  attachCardHandlers(recoGrid);

  homeView.classList.add('hidden');
  detailView.classList.remove('hidden');
  window.scrollTo({ top: 0, behavior: 'smooth' });
  history.pushState({ movieId }, '', `#movie=${movieId}`);
}

backBtn.addEventListener('click', () => {
  detailView.classList.add('hidden');
  homeView.classList.remove('hidden');
  history.pushState({}, '', '#');
});

window.addEventListener('popstate', () => {
  const hash = location.hash;
  const match = hash.match(/movie=(\d+)/);
  if (match) {
    openMovie(match[1]);
  } else {
    detailView.classList.add('hidden');
    homeView.classList.remove('hidden');
  }
});

// ---------------- init ----------------
(async function init() {
  await initGenreFilter();
  const hashMatch = location.hash.match(/movie=(\d+)/);
  if (hashMatch) {
    openMovie(hashMatch[1]);
  } else {
    loadTrending();
  }
})();
