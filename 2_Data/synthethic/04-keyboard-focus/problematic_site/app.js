const cards=[...document.querySelectorAll('.task-card')];
cards.forEach(card=>card.addEventListener('click',()=>document.querySelector('#drawer').classList.add('open')));
document.querySelector('#close-drawer').addEventListener('click',()=>document.querySelector('#drawer').classList.remove('open'));
document.querySelectorAll('.filter-chip').forEach(chip=>chip.addEventListener('click',()=>{document.querySelectorAll('.filter-chip').forEach(c=>c.classList.remove('active'));chip.classList.add('active');const filter=chip.dataset.filter;cards.forEach(card=>card.hidden=filter==='mine'?card.dataset.owner!=='me':filter==='high'?card.dataset.priority!=='high':false)}));
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));tab.classList.add('active')}));
document.querySelectorAll('.add-card,.fake-button,.column-menu').forEach(item=>item.addEventListener('click',()=>item.classList.toggle('selected')));

