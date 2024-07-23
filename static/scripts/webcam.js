export const webcam = () => {
    console.log("Webcam function called!");

    let video = document.querySelector("#videoElement");
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');

    let stream = null;

    startButton.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            startButton.disabled = true;
            stopButton.disabled = false;
        } catch (err) {
            console.error('Error accessing webcam: ', err);
        }
    });


    stopButton.addEventListener('click', () => {
        stream.getTracks().forEach(track => track.stop());
        startButton.disabled = false;
        stopButton.disabled = true;
    });

};