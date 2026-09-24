js_get_observation_and_actions = """
    domain => {
        function getXPath(element) {
            if (element === document.body) {
                return '/html/' + element.tagName.toLowerCase();
            }
            let ix = 1,
                siblings = element.parentNode.childNodes;

            for (let i = 0, l = siblings.length; i < l; i++) {
                let sibling = siblings[i];
                if (sibling === element) {
                    return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix) + ']';
                } else if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                    ix++;
                }
            }
        }
        function getElementByXpath(xpath){
            return document.evaluate(xpath,document).iterateNext();
        }
        function getElementAttributes(element, elementActionType = 1) {
            let rect = element.getBoundingClientRect();
            let attributes = {
                top: rect.top,
                right: rect.right,
                bottom: rect.bottom,
                left: rect.left,
                width: rect.width,
                height: rect.height
            };
            attributes['outerHTML'] = element.outerHTML
            attributes['xpath'] = getXPath(element)
            attributes['tagName'] = element.tagName
            attributes['actiontype'] = elementActionType
            attributes['innerText'] = element.innerText
            return attributes;
        }

        function checkIsCovered(element) {
            let rect = element.getBoundingClientRect();
            let x = rect.left + rect.width / 2;
            let y = rect.top + rect.height / 2;
            let elements = document.elementsFromPoint(x, y);
            for (let i = 0; i < elements.length; i++) {
                let item = elements[i];
                if (element === item) {
                    return false;
                }
                if (element.contains(item)) {
                    continue;
                } else {
                    return true;
                }
            }
        }

        function checkHref(element, domain) {
            // phoenix, examxx +
            // if (!element.hasAttribute('href')  && !element.innerText && !element.outerHTML.includes('icon')) {
            //     return false;
            // }  
            // phoenix, examxx -
            let href = element.href;
            if (href.includes('mailto:')) {
                return -1;
            }
            if (href === null || href.includes('@') || href.endsWith('.jpg') || href.endsWith('type=xml') || href.endsWith('wsdl') || href.includes('api-docs') ) {
                return 0;
            }
            if (href === "javascript:void(0)" || href === '') { // petclinic
                return 1;
            }
            // return href.includes(domain);
            if (href.includes(domain)) {
                return 1;
            } else {
                return -1;
            }
        }

        function checkIsValidElement(element) {
            // default
            // let default_conditions = !element | !element.enabled |element.disabled | checkIsCovered(element);  Todo Add more
            if ((element.offsetHeight === 0 || element.offsetWidth === 0) && element.tagName === 'FORM') {
              return 1;
            }
            let style = window.getComputedStyle(element);
            if (
                style.display === 'none' ||
                style.visibility === 'hidden' ||
                style.visibility === 'collapse' ||
                Number(style.opacity || 1) <= 0.01 ||
                style.pointerEvents === 'none'
            ) {
                return false;
            }
            return !(!element || element.disabled || checkIsCovered(element));
        }

        function checkIsAction(element) {
            let elemActionType = 0;
            // check by tag name
            let elemTagName = element.tagName.toLowerCase();
            switch (elemTagName) {
                case 'a':
                case 'area':
                    elemActionType = checkHref(element, domain);
                    break;
                case 'button':
                    elemActionType = 1;
                    break;
                case 'select':
                    elemActionType = 3;
                    break;
                case 'input':
                    if (!element.readOnly) {
                        let clickTypes = ['submit', 'checkbox', 'button', 'radio'];
                        let typeTypes = ['text', 'password', 'number', 'email', 'date', 'search'];
                        if (clickTypes.includes(element.type)) {
                            elemActionType = 1;
                        } else if (typeTypes.includes(element.type)) {
                            elemActionType = 2;
                        }
                        // else{
                        //     elemActionType = 0;
                        // }
                    }
                    break;
                case 'textarea':
                    elemActionType = 2;
                    break;
                case 'form':
                    elemActionType = 4;
                    break;
            }
            return elemActionType;
        }

        let leftTagName = ['DIV', 'SPAN', 'LI', 'LABEL', 'H4', 'H1', 'I'];

        function f_zzh(element) {
            return window.getComputedStyle(element)['cursor'] === 'pointer';
        }

        function isInViewport(element) {
            let rect = element.getBoundingClientRect();
            return (
                rect.top >= 0 &&
                rect.left >= 0 &&
                rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
                rect.right <= (window.innerWidth || document.documentElement.clientWidth)
            );
        }

        function getActionSpace() {
            let bodyElement = document.body;
            let actionsXPATH = new Set();
            let actionSpace = [];
            let potentialActions = [];
            let notInViewport = [];
            let queue = [];
            queue.push(bodyElement);

            while (queue.length > 0) {
                let currentElement = queue.shift();
                if (currentElement.tagName === 'IFRAME') { continue; }
                let isValid = checkIsValidElement(currentElement);
                if (currentElement.tagName === 'FORM') {
                    let currentElementActionType = checkIsAction(currentElement);
                    if (isValid && currentElementActionType > 0){
                        let actionAttrubutes = getElementAttributes(currentElement, currentElementActionType);
                        actionSpace.push(actionAttrubutes);
                        actionsXPATH.add(actionAttrubutes['xpath']);
                    }
                    // a
                    let aElements = Array.from(currentElement.getElementsByTagName('a'));
                    for (let i = 0; i < aElements.length; i++) {
                        let aElement = aElements[i];
                        if (checkIsValidElement(aElement) && checkIsAction(aElement) === 1) {
                            let a_actionAttrubutes = getElementAttributes(aElement, 1);
                            actionSpace.push(a_actionAttrubutes);
                            actionsXPATH.add(a_actionAttrubutes['xpath']);
                        }
                    }
                    // li
                    let liElements = Array.from(currentElement.getElementsByTagName('li'));
                    for (let i = 0; i < liElements.length; i++) {
                        let liElement = liElements[i];
                        if (checkIsValidElement(liElement) && liElement.hasAttribute('class') && liElement.getAttribute('class').includes("dynamictable-captionaction")) {
                            let li_actionAttrubutes = getElementAttributes(liElement, 1);
                            actionSpace.push(li_actionAttrubutes);
                            actionsXPATH.add(li_actionAttrubutes['xpath']);
                        }
                    }
                    continue;
                }
                if (isValid) {
                    let currentElementActionType = checkIsAction(currentElement);
                    if (currentElement.tagName === 'A') {
                        console.log(currentElement);
                        console.log(currentElementActionType);
                    }
                    if (currentElementActionType > 0) {
                        isValid = false;
                        if (isInViewport(currentElement)) {
                            let actionAttrubutes = getElementAttributes(currentElement, currentElementActionType);
                            actionSpace.push(actionAttrubutes);
                            actionsXPATH.add(actionAttrubutes['xpath']);
                        } else {
                            notInViewport.push(currentElement);
                        }
                    }
                    if (currentElementActionType == -1) { continue; }
                }
                if (currentElement.children.length > 0) {
                    for (let i = 0; i < currentElement.children.length; i++) {
                        queue.push(currentElement.children[i]);
                    }
                } else {
                    if (isValid && leftTagName.includes(currentElement.tagName)) {
                        if (f_zzh(currentElement)) {
                            parentXPATH = getXPath(currentElement.parentNode);
                            if (!actionsXPATH.has(parentXPATH)) {
                                let actionAttrubutes = getElementAttributes(currentElement, 1);
                                actionSpace.push(actionAttrubutes);
                                actionsXPATH.add(actionAttrubutes['xpath']);
                            }
                        } else if (currentElement.tagName === 'LI' && currentElement.hasAttribute('class') && currentElement.getAttribute('class').includes("dynamictable-captionaction")) {
                            let actionAttrubutes = getElementAttributes(currentElement, 1);
                            actionSpace.push(actionAttrubutes);
                            actionsXPATH.add(actionAttrubutes['xpath']);
                        }
                        else {
                            if (checkIsValidElement(currentElement) && (currentElement.offsetHeight != 0 && currentElement.offsetWidth != 0)){
                                let actionAttrubutes = getElementAttributes(currentElement, 0);
                                potentialActions.push(actionAttrubutes);
                            }
                        }
                    }
                }
            }
            for (let i = 0; i < notInViewport.length; i++) {
                let elem = notInViewport[i];
                elem.scrollIntoView();
                if (checkIsValidElement(elem) && isInViewport(elem)) {
                    let currentElementActionType = checkIsAction(elem);
                    if (currentElementActionType != -1) {
                        let actionAttrubutes = getElementAttributes(elem, currentElementActionType);
                        actionSpace.push(actionAttrubutes);
                        actionsXPATH.add(actionAttrubutes['xpath']);
                    }
                }
            }
            let liveActionSpace = [];
            for (let i = 0; i < actionSpace.length; i++) {
                let element = getElementByXpath(actionSpace[i]['xpath']);
                if (!element) {
                    continue;
                }
                let rect = element.getBoundingClientRect();
                let attributes = {
                    top: rect.top,
                    right: rect.right,
                    bottom: rect.bottom,
                    left: rect.left,
                    width: rect.width,
                    height: rect.height
                };
                Object.assign(actionSpace[i], attributes);
                liveActionSpace.push(actionSpace[i]);
            }
            actionSpace = liveActionSpace;
            return {
                observation: document.documentElement.outerHTML,
                actionSpace: actionSpace,
                potentialActions: potentialActions
            };
        }
        return getActionSpace();
    }
    """
js_form_action = """
    formXpath => {
        function getXPath(element) {
            if (element === document.body) {
                return '/html/' + element.tagName.toLowerCase();
            }
            var ix = 1, 
                siblings = element.parentNode.childNodes; 
        
            for (var i = 0, l = siblings.length; i < l; i++) {
                var sibling = siblings[i];
                if (sibling === element) {
                    return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix) + ']';
                } else if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                    ix++;
                }
            }
        }
        function checkIsCovered(element) {
            let rect = element.getBoundingClientRect();
            let x = rect.left + rect.width / 2;
            let y = rect.top + rect.height / 2;
            let elements = document.elementsFromPoint(x, y);
            for (let i = 0; i < elements.length; i++) {
                let item = elements[i];
                if (element === item) {
                    return false;
                }
                if (element.contains(item)) {
                    continue;
                } else {
                    return true;
                }
            }
        }
        function checkHref(element, domain) {
            if (!element.hasAttribute('href')) {
                return false;
            }
            // let href = element.getAttribute('href');
            let href = element.href;
            // if (!href || href.includes('@')) {
            //    return false;
            // }
            if (href === null || href.includes('@')) {
                return false;
            }
            if (href === "javascript:void(0)") {
                return true;
            }
            return href.includes(domain);
        }
        
        function checkIsAction(element) {
            let elemActionType = 0;
            // default
            // let default_conditions = !element | !element.enabled |element.disabled | checkIsCovered(element);  Todo Add more
            let default_conditions = !element | element.disabled | checkIsCovered(element);
            //if (element.height === 0 || element.width === 0) {
            //  return 0
            //}
            if (default_conditions) {
                return elemActionType;
            }
            // check by tag name
            let elemTagName = element.tagName.toLowerCase();
            switch (elemTagName) {
                case 'button':
                    elemActionType = 1;
                    break;
                case 'select':
                    elemActionType = 3;
                    break;
                case 'input':
                    if (!element.readOnly) {
                        let clickTypes = ['submit', 'checkbox', 'button', 'radio'];
                        let typeTypes = ['text', 'password', 'number', 'email', 'date', 'search'];
                        if (clickTypes.includes(element.type)) {
                            elemActionType = 1;
                        } else if (typeTypes.includes(element.type)) {
                            elemActionType = 2;
                        }
                        // else{
                        //     elemActionType = 0;
                        // }
                    }
                    break;
                case 'textarea':
                    elemActionType = 2;
                    break;
                case 'form':
                    elemActionType = 4;
                    break;
            }
            return elemActionType;
        }
        function getFormActionAttributes(element, elementActionType = 1) {
            let attributes = {};
            attributes['outerHTML'] = element.outerHTML
            attributes['xpath'] = getXPath(element)
            attributes['actiontype'] = elementActionType
            return attributes;
        }
        function getFormActions(formXPATH) {
            let formAction = document.evaluate(formXPATH, document).iterateNext()
            let clickActions = [];
            let typeActions = [];
            let selectActions = [];
            let queue = [];
            queue.push(formAction);
            while (queue.length > 0) {
                let currentElement = queue.shift();
                let currentElementActionType = checkIsAction(currentElement)
                if (currentElementActionType === 1) {
                    clickActions.push(getFormActionAttributes(currentElement, currentElementActionType));
        
                } else if (currentElementActionType === 2) {
                    typeActions.push(getFormActionAttributes(currentElement, currentElementActionType));
                }else if (currentElementActionType === 3) {
                    selectActions.push(getFormActionAttributes(currentElement, currentElementActionType));
                }
                
                for (let i = 0; i < currentElement.children.length; i++) {
                    queue.push(currentElement.children[i]);
                }
            }
            return {
                clickActions: clickActions,
                typeActions: typeActions,
                selectActions: selectActions
            };
        }
        return getFormActions(formXpath);
    }
    """
js_get_html = """
    () => {
        return document.documentElement.outerHTML;
    }
"""
