$(document).ready(function(){
        $('#imgInp').on('change', function() {
          var file = this.files[0];
          console.log(file);

          if (file) {
            var fileURL = URL.createObjectURL(file);
            console.log(fileURL);
          } else {
            console.log('No file selected.');
          }
          sendPhotoToBackend(file);
        });

        // Send photo to the backend
        function sendPhotoToBackend(photoData) {
          // Make a POST request to the backend endpoint
          var csrf_token = $("input[name='csrfmiddlewaretoken']").val();
          var formData = new FormData();
          formData.append('photo', photoData);
          formData.append('csrfmiddlewaretoken', csrf_token);
          $.ajax({
            url:'/process-image/',
            type: 'POST',
            headers:{'X-CSRFToken':csrf_token},
            data: formData,
            processData: false,
            contentType: false,
            success:function(data){
              console.log('Photo sent successfully', data['url']);
              $('#cropped_img').attr('src', data['url']);
              $('.image_area').css('display', 'block');
            },
            error:function(error){
              console.error('Error sending photo:', error);
            }
          });
        }
      });