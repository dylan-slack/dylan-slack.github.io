var animate, left=0, imgObj=null, flip=0;
var counter = 0;
var pointerX = -1;
var pointerY = -1;
var rotating = 360;
var doing_flip = 0;
var moving_to_side = 0;
var side = 0;
var start = 1;

var width = 0;
width = window.screen.availWidth - 100;
var inc = 0;
inc = width / 100;
var move_counter = 0;

document.onmousemove = function(event) {
    pointerX = event.pageX;
    pointerY = event.pageY;
}

function init(){
   imgObj = document.getElementById('image');
   imgObj.style.position= 'absolute';
   imgObj.style.top = '25px';
   imgObj.style.left = '-50px';
   imgObj.style.visibility='invisible';

   
    // document.addEventListener('scroll', function(e) {
        if (start == 1) {
            start = 0;
            animate = setTimeout(function(){move();},20);
        }
    // });


    document.getElementById("image").addEventListener("click", function(){
        console.log('here');
        var offsets = document.getElementById('mostrecentpaper').getBoundingClientRect();
        var top = offsets.top;
        var left = offsets.left;
        console.log(top);
        console.log(left);
    });


   // if (start == 1){
   //      move();
   // }
}

function move(){
    left = parseInt(imgObj.style.left, 10);
    // alert(window.screen.availWidth);
    // alert(window.screen.availHeight);

    if (counter <= 20) {
        imgObj.style.left = (left + 5) + 'px';
        imgObj.style.visibility='visible';
        counter = counter + 1;
        animate = setTimeout(function(){move();},20); // call move in 20msec
    } else {

        index = Math.floor(100000 * Math.random());
        // if (pointerX >= (parseInt(imgObj.style.left) + 10)) {
        if ((index < 100 && doing_flip == 0 && side == 0) || doing_flip == 1) {
            doing_flip = 1;
            if (rotating >= 0) {
                doFlip();
                animate = setTimeout(function(){move();}, 50);
            } 
        } else if ((index >= 100 && index <= 200 && moving_to_side == 0) || moving_to_side == 1) {
            moving_to_side = 1;
            if (side == 0) {
                imgObj.style.left = (left + 5) + 'px';
            }
            else {
                imgObj.style.left = (left - 5) + 'px';
            }
            new_left = parseInt(imgObj.style.left, 10);
            if (new_left < 20 || new_left > (window.screen.availWidth * 0.8)) {
                moving_to_side = 0;
                move_counter = 0;
                if (side == 0) {
                    imgObj.style.transform = "scaleX(-1)";
                    side = 1;
                } else {
                    imgObj.style.transform = "scaleX(1)";
                    side = 0;
                }
            }
            animate = setTimeout(function(){move();}, 15);
        } 
        else {
            // imgObj.style.transform = "scaleX(1)";
            animate = setTimeout(function(){move();}, 5);
        }
    }
}

function doFlip(){
    left = parseInt(imgObj.style.left, 10);
    top = parseInt(imgObj.style.top, 10);
    imgObj.style.transform = "rotate(" + rotating + "deg)";
    rotating = rotating - 24;

    if (rotating < 0) {
        doing_flip = 0;
        rotating = 360;
    }
}

function moveCloser(){
    left = parseInt(imgObj.style.left, 10);
    imgObj.style.left = (left + (pointerX - left) * 0.01) + 'px';
    imgObj.style.top = (parseInt(imgObj.style.top, 10) + (pointerY - parseInt(imgObj.style.top, 10)) * 0.1) + 'px';
    imgObj.style.visibility='visible';
}

function stop(){
   clearTimeout(animate);
}

window.onload = function() {init();};