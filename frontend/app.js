const express = require('express');
const app = express();
const router = express.Router();

const path = __dirname + '/views/';
const port = 8080;

var proxy = require('http-proxy').createProxyServer({
    ignorePath: true // Needed to prevent proxy adding a trailing '/'
});

router.use(function (req,res,next) {
  console.log('/' + req.method);
  next();
});

router.get('/', function(req,res){
  res.sendFile(path + 'index.html');
});

// Simple test
app.use('/hello', function(req, res, next) {
    proxy.web(req, res, {
        target: 'http://localhost:5000/hello',
    }, next);
});

// Proxy request to actual API server
app.use('/sudoku', function(req, res, next) {
    proxy.web(req, res, {
        target: 'http://localhost:5000/sudoku',
    }, next);
});

app.use(express.static(path));
app.use('/', router);

app.listen(port, function () {
  console.log('Example app listening on port 8080!')
})
