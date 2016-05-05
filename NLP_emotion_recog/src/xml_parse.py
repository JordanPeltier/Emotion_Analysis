from xml.dom.minidom import parse
import xml.dom.minidom


class Word:
    def __init__(self, word_xml):
        if word_xml.hasAttribute("punc"):
            self.meeting_id = word_xml.getAttribute("nite:id")[0:7]
            self.participant_id = word_xml.getAttribute("nite:id")[8]
            self.word_id = word_xml.getAttribute("nite:id")[10:]
            self.start_time = float(word_xml.getAttribute("starttime"))
            self.end_time = float(word_xml.getAttribute("endtime"))
            self.punc = True
            self.content = word_xml.childNodes[0].data
        else:
            self.meeting_id = word_xml.getAttribute("nite:id")[0:7]
            self.participant_id = word_xml.getAttribute("nite:id")[8]
            self.word_id = word_xml.getAttribute("nite:id")[10:]
            self.start_time = float(word_xml.getAttribute("starttime"))
            self.end_time = float(word_xml.getAttribute("endtime"))
            self.punc = False
            self.content = word_xml.childNodes[0].data

    def __str__(self):
        input_string = ["meeting_id= " + self.meeting_id,
                        "participant_id = " + self.participant_id,
                        "word_id = " + str(self.word_id),
                        "start_time = " + str(self.start_time),
                        "end_time = " + str(self.end_time),
                        "punc = " + str(self.punc),
                        "content = " + self.content]

        return "\n".join(input_string)




def parse_xml(path):
    # Open XML document using minidom parser
    dom_tree = xml.dom.minidom.parse(path)
    collection = dom_tree.documentElement
    words_xml = collection.getElementsByTagName("w")
    words = list()
    for word in words_xml:
        words.append(Word(word))
    return words
