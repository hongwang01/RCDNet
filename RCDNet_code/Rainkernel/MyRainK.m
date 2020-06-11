function h = MyRainK(p2, long, inner, std)
siz   = (p2-1)/2;
[x,y] = meshgrid(-siz:siz,-siz:siz);
inner = round(siz*inner);
x = sign(x).*(max(abs(x),inner)-inner);
y = sign(y).*(max(abs(y),inner)-inner);
y = y/long;
arg   = -(x.*x + y.*y)/(2*std*std);

h     = exp(arg);
h(h<eps*max(h(:))) = 0;

sumh = (sum(h(:)));
if sumh ~= 0
    h  = h/sumh;
end
end