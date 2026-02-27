document.addEventListener('DOMContentLoaded', function() {
    // Example validation for session form (schedule_sessions.html)
    const form = document.querySelector('form');
    const startTimeInput = document.getElementById('start_time');
    const endTimeInput = document.getElementById('end_time');

    form.addEventListener('submit', function(event) {
        // Validate that the start time is before the end time
        const startTime = startTimeInput.value;
        const endTime = endTimeInput.value;

        if (startTime >= endTime) {
            alert('Start time must be earlier than end time.');
            event.preventDefault();
        }
    });

    function toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        const content = document.getElementById('content');
        sidebar.classList.toggle('active');
        
        // Adjust content margin
        if (sidebar.classList.contains('active')) {
            content.style.marginLeft = '250px';
        } else {
            content.style.marginLeft = '0';
            content.style.textAlign = 'center'; /* Center content horizontally */
        }
    }
});
