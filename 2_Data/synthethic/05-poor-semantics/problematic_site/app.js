const toast=document.querySelector('#toast');
document.querySelector('.newsletter form').addEventListener('submit',event=>{event.preventDefault();toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),1600)});
document.querySelectorAll('.feature button').forEach(button=>button.addEventListener('click',()=>{button.textContent=button.textContent==='Read story'?'Saved to reading list':'Read story'}));

