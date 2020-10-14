import re
import datetime
from num2words import num2words

pos = [('Bác_Hồ', 'Np'), ('là', 'V'), ('người', 'N'), ('con', 'N'), ('thứ', 'N'), ('3', 'M'), ('trong', 'E'), ('gia_đình', 'N'), ('5', 'M'), ('người', 'N'), ('và', 'Cc'), ('sinh', 'V'), ('năm', 'N'), ('02/2', 'M')]

def addToTuple(data, value, index):
  tempList = list(data[index])
  tempList[0] = value
  data[index] = tuple(tempList)

def transferYear(data, inputString, usedRegex, format, index):
  try:
    wordBeforeDate = data[index - 1][0]
    dateFormat = ''
    dateString = ''
    # If string is full date format day/month/year
    if (usedRegex is 'fullDate'):
      dateFormat = "%d" + format + "%m" + format +"%Y"
      date = datetime.datetime.strptime(inputString, dateFormat)
      dateString += "ngày " + num2words(date.day, lang='vi') + " tháng " + num2words(date.month, lang='vi') + " năm " + num2words(date.year, lang='vi')
    # If string has only day and month
    if (usedRegex is 'DayMonth'):
      dateFormat = "%d" + format + "%m"
      date = datetime.datetime.strptime(inputString, dateFormat)
      dateString += "ngày " + num2words(date.day, lang='vi') + " tháng " + num2words(date.month, lang='vi')
    # If string has only month and year
    if (usedRegex is 'MonthYear'):
      dateFormat = "%m" + format + "%Y"
      date = datetime.datetime.strptime(inputString, dateFormat)
      dateString += "ngày " + num2words(date.month, lang='vi') + " tháng " + num2words(date.year, lang='vi')
    # If the word before date string is pronouns which indicate for date
    if (wordBeforeDate == 'ngày' or wordBeforeDate == 'mùng' or wordBeforeDate == 'tháng' or wordBeforeDate == 'năm'):
      dateString = dateString.split(' ', 1)[1]
      
    addToTuple(data, dateString, index)
  except ValueError as err:
    print(err)

def transferNumberToWord(data):
  regexFullDate = r'^(([1-9]|0[1-9]|[12]\d|3[01])(\/|-|.)([1-9]|0[1-9]|1[0-2])(\/|-|.)[12]\d{3})$'
  regexDayMonth = r'^((\b([1-9]|0[1-9]|[12][0-9]|3[01])\b)(\/|-|.)([1-9]|0[1-9]|1[0-2]))$'
  regexMonthYear = r'^((\b([1-9]|0[1-9]|1[0-2])\b)(\/|-|.)[12]\d{3})$'
  usedRegex = ''

  # Split word in tuple to an array
  sentence = [element[0] for element in data]

  for word in sentence:
    index = sentence.index(word)
    date = None

    # Search Date if it is in array
    if (re.search(regexFullDate, word) is not None):
      date = re.search(regexFullDate, word)
      usedRegex = 'fullDate'
    if (re.search(regexDayMonth, word) is not None):
      date = re.search(regexDayMonth, word)
      usedRegex = 'DayMonth'
    if (re.search(regexMonthYear, word) is not None):
      date = re.search(regexMonthYear, word)
      usedRegex = 'MonthYear'

    # Convert Date to Word
    if date is not None:
      if '/' in date.group():
        transferYear(data, date.group(), usedRegex, '/', index)
      if '-' in date.group():
        transferYear(data, date.group(), '-', index)
      if '.' in date.group():
        transferYear(data, date.group(), '.', index)
    
    # Convert others number to word
    if word.isnumeric():
      tempWord = num2words(int(word), lang='vi')
      addToTuple(data, tempWord, index)
  
  return data

pos = transferNumberToWord(pos)

print(pos)

  


