const { parseDocument, DomUtils } = require('htmlparser2');

function plainText(html) {
    const document = parseDocument(html);
    DomUtils.findAll(function (element) {
        return ['script', 'style', 'pre', 'figure'].includes(element.name);
    }, document.children).forEach(DomUtils.removeElement);
    return DomUtils.innerText(document.children).replace(/\s+/g, ' ').trim();
}

hexo.extend.helper.register('seo_description', function () {
    const page = this.page;
    const custom = page.description || page.desc || page.summary;
    const content = this.is_post() || this.is_page()
        ? page.excerpt || page.content
        : '';
    const text = plainText(custom || content || this.config.description);
    return Array.from(text).length > 160
        ? Array.from(text).slice(0, 159).join('') + '…'
        : text || this.config.description;
});

hexo.extend.helper.register('article_json_ld', function (canonical, description) {
    const page = this.page;
    const article = {
        '@context': 'https://schema.org',
        '@type': 'BlogPosting',
        '@id': canonical + '#article',
        mainEntityOfPage: { '@type': 'WebPage', '@id': canonical },
        url: canonical,
        headline: plainText(page.title),
        description: description,
        author: {
            '@type': 'Person',
            name: this.config.author,
            url: this.full_url_for('about/')
        },
        datePublished: this.date_xml(page.date),
        dateModified: this.date_xml(page.updated > page.date ? page.updated : page.date),
        inLanguage: this.config.language
    };
    const image = DomUtils.findOne(function (element) {
        return element.name === 'img' && /^\/(?!\/)/.test(element.attribs.src || '');
    }, parseDocument(page.content).children);
    if (image) {
        article.image = [new URL(image.attribs.src, canonical).href];
    }
    return JSON.stringify(article).replace(/</g, '\\u003c');
});
