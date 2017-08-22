const express = require('express');
const fs = require('fs');
const Q = require('q');
const app = express();

process.chdir(__dirname + '/public/data/')

// Javascript sometimes feels like a very cruel joke – dukeofgaming Jan 4 '15 at 13:47
Number.prototype.mod = function(n) {
    return ((this%n)+n)%n;
};

app.use(express.static(__dirname + '/public'));
app.get('/ajax/viscinity', (req, res) => {
	// which_edge: 0 | -1
	function bound(dir, which_edge, remaining) { // : Array<int>
		if(remaining === 0)
			return dir;
		else {
			const deferred = Q.defer();
			fs.readdir(dir.join('/') || '.', (e, f) => {
				if(e)
					deferred.reject(e);
				else
					deferred.resolve(f);
			});
			return deferred.promise.then((files) => {
				files = files.map((f) => parseInt(f, 10)).filter((f) => !isNaN(f));
				files.sort((a, b) => a - b);
				
				const next = dir.concat([files[which_edge.mod(files.length)]]);
				return bound(next, which_edge, remaining - 1);
			});
		}
	}
	function viscinity(dir, T) {
		// for dir, T : Array<int>
		const deferred = Q.defer();
		fs.readdir(dir.join('/') || '.', (e, f) => {
			if(e)
				deferred.reject(e);
			else
				deferred.resolve(f);
		});
		const ret = deferred.promise.then((files) => {
			files = files.map((f) => parseInt(f, 10)).filter((f) => !isNaN(f));
			files.sort((a, b) => a - b);
			
			if(T.length > 0) {
				// [locally out of bounds]
				if(T[0] < files[0])
					return bound(dir, 0, T.length).then(edge => [null, edge]);
				else if(T[0] > files[files.length - 1]) 
					return bound(dir, -1, T.length).then(edge => [edge, null]);
				// [/locally out of bounds]
				else {
					for(let i = 0; i < files.length; i++) {
						if(T[0] === files[i])
							return viscinity(dir.concat([files[i]]), T.slice(1))
							       	.then((bracket) => {
							       		if(bracket[0] == null)
							       			bracket[0] = i > 0 ? bound(dir.concat([files[i - 1]]), -1, T.length - 1) : null;
							       		if(bracket[1] == null)
							       			bracket[1] = i < files.length - 1 ? bound(dir.concat([files[i + 1]]), 0, T.length - 1) : null;
							       		
						       			return Promise.all(bracket);
							       	})
						else if(T[0] < files[i])
							return Promise.all([
								bound(dir.concat([files[i - 1]]), -1, T.length - 1),
								bound(dir.concat([files[i]]), 0, T.length - 1)
							]);
					}
				}
			}
			else {
				return [null, null];
			}
		}, (e) => console.log(e) || res.status(500).end());
		return ret;
	}
	viscinity([], JSON.parse(req.query.T))
		.then((bracket) => res.send(bracket), console.log);
})
app.listen(8082);