const toast=document.querySelector('#toast');
document.querySelectorAll('a.vague,button.vague').forEach(control=>control.addEventListener('click',event=>{event.preventDefault();toast.textContent=control.textContent.trim()||'Opened';toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),1200)}));
document.querySelector('#search-button').addEventListener('click',()=>{const term=document.querySelector('#search').value.toLowerCase();document.querySelectorAll('.resource').forEach(card=>card.classList.toggle('filtered',term&&!card.dataset.title.includes(term)))});
