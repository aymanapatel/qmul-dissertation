const toast=document.querySelector('#toast');
document.querySelectorAll('.choice').forEach(choice=>choice.addEventListener('click',()=>{document.querySelectorAll('.choice').forEach(c=>c.classList.remove('selected'));choice.classList.add('selected');choice.querySelector('input').checked=true;document.querySelector('#format-summary').textContent=choice.dataset.value==='Video'?'Video call':'At the clinic'}));
document.querySelector('#booking-form').addEventListener('submit',event=>{event.preventDefault();toast.textContent='Appointment details saved';toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),1800)});

