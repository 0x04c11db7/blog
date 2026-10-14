const { parse } = require('hexo-front-matter');

hexo.extend.filter.register('before_generate', function () {
    // Incremental builds can retain old mtime-based dates in the database.
    return Promise.all(hexo.model('Post').filter(function (post) {
        return !parse(post.raw).updated && post.updated.valueOf() !== post.date.valueOf();
    }).map(function (post) {
        return hexo.model('Post').updateById(post._id, { updated: post.date.toDate() });
    }));
});
