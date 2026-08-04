const toast=document.querySelector('#toast');
document.querySelector('#edit-budget').addEventListener('click',()=>{toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),1500)});
document.querySelector('#all').addEventListener('click',event=>{event.target.textContent=event.target.textContent==='See all'?'Show less':'See all';document.querySelector('#transaction-list').classList.toggle('expanded')});
