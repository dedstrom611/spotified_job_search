// submit.js

$(document)
.ready(function() {

$('#select_title')
  .dropdown({
    allowAdditions: true,
    fullTextSearch: true

 })
;

$('#select_states')
  .dropdown({
    fullTextSearch: true

 })
;

$('.ui.form')
  .form({
    fields: {
      job_title     : 'empty',
    }
  })
;
});
