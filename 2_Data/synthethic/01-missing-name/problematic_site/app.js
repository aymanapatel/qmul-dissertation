const toast=document.querySelector('#toast');
function flash(message){toast.textContent=message;toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),1400)}
document.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>flash('Library updated')));
document.querySelector('#asset-search').addEventListener('input',event=>{const term=event.target.value.toLowerCase();document.querySelectorAll('.asset').forEach(asset=>asset.hidden=!asset.dataset.title.includes(term))});

