// submit.js

$(document)
.ready(function() {

$('#job_title')
  .dropdown({
    allowAdditions: true,
    fullTextSearch: true

 })
;

$('#states')
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
