function getRandom(a,b) {
	var rand_num = (Math.random()*b) + a;
	return String.fromCharCode(rand_num);
}