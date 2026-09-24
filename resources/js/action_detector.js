const selectors = arguments[0];
const textDetectModeInput = arguments[1];
const textDetectMode = Object.freeze({
  INNER_TEXT: 0,
  LABEL: 1
});
const elements = selectors
    .flatMap(selector => Array.from(document.querySelectorAll(selector)))
return elements.map(elem => ({
    xpath: generateXpath(elem),
    visible: isVisible(elem),
    text: getText(elem)
}));

function isVisible(elem) {
    if (!(elem instanceof Element)) {
        return false;
    }
    const style = getComputedStyle(elem);
    if (style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse' || Number(style.opacity || 1) <= 0.01) {
        return false;
    }
    const rect = elem.getBoundingClientRect();
    if (!rect || rect.width <= 1 || rect.height <= 1) {
        return false;
    }
    if (rect.bottom < 0 || rect.right < 0 || rect.top > (window.innerHeight || document.documentElement.clientHeight) || rect.left > (window.innerWidth || document.documentElement.clientWidth)) {
        return false;
    }
    const centerX = Math.max(0, Math.min((window.innerWidth || document.documentElement.clientWidth) - 1, rect.left + rect.width / 2));
    const centerY = Math.max(0, Math.min((window.innerHeight || document.documentElement.clientHeight) - 1, rect.top + rect.height / 2));
    const topElem = document.elementFromPoint(centerX, centerY);
    if (!topElem) {
        return true;
    }
    if (topElem === elem || elem.contains(topElem) || topElem.contains(elem)) {
        return true;
    }
    return true;
}

function generateXpath(element) {
    let tag = element.tagName.toLowerCase();
//    if (element.id) {
//        return `//${tag}[@id="${element.id}"]`
//    } else
    if (element === document.documentElement) {
        return '/html'
    } else {
        const sameTagSiblings = Array.from(element.parentNode.childNodes)
            .filter(e => e.nodeName === element.nodeName)
        const idx = sameTagSiblings.indexOf(element)

        return `${generateXpath(element.parentNode)}/${tag}${sameTagSiblings.length > 1 ? `[${idx + 1}]` : ''}`
    }
}

function getText(element) {
    if (textDetectModeInput === textDetectMode.INNER_TEXT) {
        return element.innerText
    }
    else if (textDetectModeInput === textDetectMode.LABEL) {
        const label = document.querySelector(`label[for="${element.id}"]`)
        if (label) {
            return label.innerText;
        }
        else {
            return "";
        }
    }
    else {
        return "";
    }
}
