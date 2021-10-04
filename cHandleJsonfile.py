import json

def ReadElementsFromJsonfile(nstrPath, nstrSectionName, ndicElements):
    with open(nstrPath , 'r', encoding="utf-8") as m_jsonfile:
        m_dicElements_section = json.load(m_jsonfile)
        m_dicElements = m_dicElements_section[nstrSectionName]
    for loopkeys in ndicElements.keys():
        if loopkeys in m_dicElements:
            ndicElements[loopkeys] = m_dicElements[loopkeys]
def WriteElementsToJsonfile(nstrPath ,nstrSectionName, ndicElements):
    with open(nstrPath , 'r', encoding="utf-8") as m_jsonfile:
        m_dicElements_section = json.load(m_jsonfile)
    with open(nstrPath , 'w', encoding="utf-8") as m_jsonfile:
        m_dicElements_section[nstrSectionName] = ndicElements
        m_dicElements = json.dumps(m_dicElements_section)
        m_jsonfile.write(m_dicElements)