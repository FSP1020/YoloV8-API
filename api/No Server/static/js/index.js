function myFunction2() {
    $.post('/api', {"value":"TEST Success"}, function(response) {
        console.log(response);  // Response from the server
    })
    .fail(function(error) {
        console.error('Error:', error);
    });
}