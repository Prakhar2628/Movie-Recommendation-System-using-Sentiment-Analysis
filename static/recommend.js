$(function() {
  // Enable/disable search button based on input
  const source = document.getElementById('autoComplete');
  if (source) {
    const inputHandler = function(e) {
      if(e.target.value === ""){
        $('.movie-button').attr('disabled', true);
      } else {
        $('.movie-button').attr('disabled', false);
      }
    };
    source.addEventListener('input', inputHandler);
  }

  $('.movie-button').on('click', function(){
    var my_api_key = typeof TMDB_API_KEY !== 'undefined' && TMDB_API_KEY ? TMDB_API_KEY : '';
    var title = $('#autoComplete').val().trim();
    if (title === "") {
      $('.results').css('display','none');
      $('.fail').css('display','block');
      $('#mood, #time, #time-section, #category, #trending').show();
    } else {
      load_details(my_api_key, title);
    }
  });
});

// Clicking a recommended movie card
function recommendcard(e){
  var my_api_key = typeof TMDB_API_KEY !== 'undefined' && TMDB_API_KEY ? TMDB_API_KEY : '';
  var title = e.getAttribute('title');
  if (title) {
    load_details(my_api_key, title);
  }
}

// Direct search TMDB for the movie title to get its ID
function load_details(my_api_key, title){
  $("#loader").fadeIn();
  $.ajax({
    type: 'GET',
    url: 'https://api.themoviedb.org/3/search/movie?api_key=' + my_api_key + '&query=' + encodeURIComponent(title),
    success: function(movie){
      if(!movie.results || movie.results.length < 1){
        $('.fail').css('display','block');
        $('.results').css('display','none');
        $('#mood, #time, #time-section, #category, #trending').show();
        $("#loader").delay(500).fadeOut();
      } else {
        $('.fail').css('display','none');
        var movie_id = movie.results[0].id;
        var movie_title = movie.results[0].original_title;
        movie_recs(movie_title, movie_id, my_api_key);
      }
    },
    error: function(){
      alert('Invalid Request');
      $("#loader").delay(500).fadeOut();
    },
  });
}

// Get similar movies from Flask ML model or TMDB fallback
function movie_recs(movie_title, movie_id, my_api_key){
  $.ajax({
    type: 'POST',
    url: "/similarity",
    data: {'name': movie_title},
    success: function(recs){
      if(recs == "Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies"){
        // Fallback to TMDB recommendations if title not in local CSV database
        $.ajax({
          type: 'GET',
          url: 'https://api.themoviedb.org/3/movie/' + movie_id + '/recommendations?api_key=' + my_api_key,
          success: function(tmdb_recs) {
            var arr = (tmdb_recs.results || []).slice(0, 10).map(function(m){ return m.title; });
            if (arr.length === 0) {
              arr = ["Inception", "Interstellar", "The Dark Knight", "Avatar", "Titanic"];
            }
            get_movie_details(movie_id, my_api_key, arr, movie_title);
          }
        });
      } else {
        $('.fail').css('display','none');
        var arr = recs.split('---');
        get_movie_details(movie_id, my_api_key, arr, movie_title);
      }
    },
    error: function(){
      alert("Error getting recommendations");
      $("#loader").delay(500).fadeOut();
    },
  });
}

// Fetch full movie details from TMDB
function get_movie_details(movie_id, my_api_key, arr, movie_title) {
  $.ajax({
    type: 'GET',
    url: 'https://api.themoviedb.org/3/movie/' + movie_id + '?api_key=' + my_api_key,
    success: function(movie_details){
      show_details(movie_details, arr, movie_title, my_api_key, movie_id);
    },
    error: function(){
      alert("API Error!");
      $("#loader").delay(500).fadeOut();
    },
  });
}

// Build and send all data to Flask /recommend endpoint
function show_details(movie_details, arr, movie_title, my_api_key, movie_id){
  var imdb_id    = movie_details.imdb_id;
  var poster     = movie_details.poster_path ? ('https://image.tmdb.org/t/p/original' + movie_details.poster_path) : 'https://via.placeholder.com/300x450/120924/8c52ff?text=🎬';
  var overview   = movie_details.overview || 'No overview available.';
  var genres     = movie_details.genres || [];
  var rating     = movie_details.vote_average || 0;
  var vote_count = movie_details.vote_count || 0;
  var release_date = movie_details.release_date ? new Date(movie_details.release_date) : new Date();
  var runtime    = parseInt(movie_details.runtime) || 120;
  var status     = movie_details.status || 'Released';

  var genre_list = [];
  for (var genre in genres){ genre_list.push(genres[genre].name); }
  var my_genre = genre_list.join(", ");

  if(runtime % 60 === 0){
    runtime = Math.floor(runtime/60) + " hour(s)";
  } else {
    runtime = Math.floor(runtime/60) + " hour(s) " + (runtime%60) + " min(s)";
  }

  var arr_poster  = get_movie_posters(arr, my_api_key);
  var movie_cast  = get_movie_cast(movie_id, my_api_key);
  var ind_cast    = get_individual_cast(movie_cast, my_api_key);

  // ── Fetch TMDB reviews & run sentiment analysis ──────
  var raw_reviews = get_tmdb_reviews(movie_id, my_api_key);
  var tmdb_review_texts = JSON.stringify(
    raw_reviews.map(function(r){ return r.content.substring(0, 600); })
  );

  var analyzed_reviews = {};
  $.ajax({
    type: 'POST',
    url: '/analyze_sentiment',
    data: { 
      reviews: tmdb_review_texts,
      title: movie_title,
      overview: overview
    },
    async: false,
    success: function(res) { analyzed_reviews = res; },
    error:   function()    { analyzed_reviews = {}; }
  });

  var details = {
    'title':          movie_title,
    'cast_ids':       JSON.stringify(movie_cast.cast_ids),
    'cast_names':     JSON.stringify(movie_cast.cast_names),
    'cast_chars':     JSON.stringify(movie_cast.cast_chars),
    'cast_profiles':  JSON.stringify(movie_cast.cast_profiles),
    'cast_bdays':     JSON.stringify(ind_cast.cast_bdays),
    'cast_bios':      JSON.stringify(ind_cast.cast_bios),
    'cast_places':    JSON.stringify(ind_cast.cast_places),
    'imdb_id':        imdb_id,
    'poster':         poster,
    'genres':         my_genre,
    'overview':       overview,
    'rating':         rating,
    'vote_count':     vote_count.toLocaleString(),
    'release_date':   release_date.toDateString().split(' ').slice(1).join(' '),
    'runtime':        runtime,
    'status':         status,
    'rec_movies':     JSON.stringify(arr),
    'rec_posters':    JSON.stringify(arr_poster),
    'analyzed_reviews': JSON.stringify(analyzed_reviews),
  };

  $.ajax({
    type: 'POST',
    data: details,
    url: "/recommend",
    dataType: 'html',
    complete: function(){
      $("#loader").delay(500).fadeOut();
    },
    success: function(response) {
      $('.results').html(response);
      $('.results').css('display','block');
      $('#mood, #time, #time-section, #category, #trending').hide();
      $('#autoComplete').val('');
      $(window).scrollTop(0);
    }
  });
}

// Fetch TMDB reviews for sentiment analysis
function get_tmdb_reviews(movie_id, my_api_key) {
  var reviews_list = [];
  $.ajax({
    type: 'GET',
    url: 'https://api.themoviedb.org/3/movie/' + movie_id + '/reviews?api_key=' + my_api_key + '&language=en-US&page=1',
    async: false,
    success: function(data) {
      var results = data.results || [];
      for (var i = 0; i < Math.min(results.length, 10); i++) {
        reviews_list.push({ content: results[i].content, author: results[i].author });
      }
    }
  });
  return reviews_list;
}

// Get individual cast bios
function get_individual_cast(movie_cast, my_api_key) {
  var cast_bdays  = [];
  var cast_bios   = [];
  var cast_places = [];
  for(var cast_id in movie_cast.cast_ids){
    $.ajax({
      type: 'GET',
      url: 'https://api.themoviedb.org/3/person/' + movie_cast.cast_ids[cast_id] + '?api_key=' + my_api_key,
      async: false,
      success: function(cast_details){
        var bd = cast_details.birthday ? (new Date(cast_details.birthday)).toDateString().split(' ').slice(1).join(' ') : 'Unknown';
        cast_bdays.push(bd);
        cast_bios.push(cast_details.biography || 'No biography available.');
        cast_places.push(cast_details.place_of_birth || 'Unknown');
      }
    });
  }
  return {cast_bdays: cast_bdays, cast_bios: cast_bios, cast_places: cast_places};
}

// Get top 10 cast for a movie
function get_movie_cast(movie_id, my_api_key){
  var cast_ids      = [];
  var cast_names    = [];
  var cast_chars    = [];
  var cast_profiles = [];

  $.ajax({
    type: 'GET',
    url: "https://api.themoviedb.org/3/movie/" + movie_id + "/credits?api_key=" + my_api_key,
    async: false,
    success: function(my_movie){
      var cast_list = my_movie.cast || [];
      var top_count = Math.min(cast_list.length, 10);
      for(var i = 0; i < top_count; i++){
        cast_ids.push(cast_list[i].id);
        cast_names.push(cast_list[i].name);
        cast_chars.push(cast_list[i].character || 'Actor');
        var p = cast_list[i].profile_path ? ("https://image.tmdb.org/t/p/w185" + cast_list[i].profile_path) : "https://via.placeholder.com/140x180/120924/8c52ff?text=👤";
        cast_profiles.push(p);
      }
    },
    error: function(){ $("#loader").delay(500).fadeOut(); }
  });

  return {cast_ids: cast_ids, cast_names: cast_names, cast_chars: cast_chars, cast_profiles: cast_profiles};
}

// Fetch posters for all recommended movies
function get_movie_posters(arr, my_api_key){
  var arr_poster_list = [];
  for(var m in arr) {
    $.ajax({
      type: 'GET',
      url: 'https://api.themoviedb.org/3/search/movie?api_key=' + my_api_key + '&query=' + encodeURIComponent(arr[m]),
      async: false,
      success: function(m_data){
        if(m_data.results && m_data.results.length > 0 && m_data.results[0].poster_path){
          arr_poster_list.push('https://image.tmdb.org/t/p/w342' + m_data.results[0].poster_path);
        } else {
          arr_poster_list.push('https://via.placeholder.com/160x240/120924/8c52ff?text=🎬');
        }
      },
      error: function(){ arr_poster_list.push('https://via.placeholder.com/160x240/120924/8c52ff?text=🎬'); }
    });
  }
  return arr_poster_list;
}

// Watch Youtube Trailer
function watchTrailer(title) {
  var my_api_key = typeof TMDB_API_KEY !== 'undefined' && TMDB_API_KEY ? TMDB_API_KEY : '';
  $.ajax({
    type: 'GET',
    url: 'https://api.themoviedb.org/3/search/movie?api_key=' + my_api_key + '&query=' + encodeURIComponent(title),
    success: function(r) {
      if(r.results && r.results.length > 0) {
        var id = r.results[0].id;
        $.ajax({
          type: 'GET',
          url: 'https://api.themoviedb.org/3/movie/' + id + '/videos?api_key=' + my_api_key,
          success: function(v) {
            var trailer = (v.results || []).find(function(item){ return item.type === 'Trailer' && item.site === 'YouTube'; }) || v.results[0];
            if(trailer && trailer.key) {
              window.open('https://www.youtube.com/watch?v=' + trailer.key, '_blank');
            } else {
              window.open('https://www.youtube.com/results?search_query=' + encodeURIComponent(title + ' trailer'), '_blank');
            }
          }
        });
      }
    }
  });
}
