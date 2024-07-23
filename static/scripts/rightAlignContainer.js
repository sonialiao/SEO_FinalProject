export const rightAlignContainer = () =>{
    var image = document.getElementById('aslChart');
    var textarea = document.getElementById('transcriptionTextarea');
    textarea.style.width = image.offsetWidth + 'px';
};