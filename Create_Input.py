import pandas as pd

xlsx = pd.ExcelFile('past_data.xlsx')
df_2014 = pd.read_excel(xlsx, 'Sheet1')
df_2015 = pd.read_excel(xlsx, 'Sheet2')
df_2016 = pd.read_excel(xlsx, 'Sheet3')
df_2017 = pd.read_excel(xlsx, 'Sheet4')

mcp_2014=df_2014['MCP']
mcp_2015=df_2015['MCP']
mcp_2016=df_2016['MCP']
mcp_2017=df_2017['MCP']

mcp_past=pd.concat([mcp_2014,mcp_2015,mcp_2016],ignore_index=True)

w, h1 ,h2 = 10, int((len(mcp_past)+1)/96)-10 , int((len(mcp_2017)+1)/96)-10

input_matrix = [[0 for x in range(w)] for y in range(h1)]  #A matrix which contains data of past 10 days. Row 0 has day 0 to day 9 and so on.
test_input_matrix = [[0 for x in range(w)] for y in range(h2)]
input_data=[]
test_input_data=[]
true_output_list=[]
test_true_output_list=[]
										#Row 0 has data of 11th day and so on.
for i in range(len(mcp_past.values)): #Get values in time slot 00:00 - 00:15
	if i%96==0:
		input_data.append(int(mcp_past[i])) #Removes the indexes and converts datatype from Pandas series to numpy array, which is compatible with TF.

for i in range(len(input_data)-10):
	for j in range(10):
		input_matrix[i][j]=input_data[i+j]

for i in range(len(input_matrix)):
	true_output_list.append(input_data[i+10])


for i in range(len(mcp_2017.values)): #Get values in time slot 00:00 - 00:15
	if i%96==0:
		test_input_data.append(int(mcp_2017[i])) #Removes the indexes and converts datatype from Pandas series to numpy array, which is compatible with TF.

for i in range(len(test_input_data)-10):
	for j in range(10):
		test_input_matrix[i][j]=test_input_data[i+j]

for i in range(len(test_input_matrix)):
	test_true_output_list.append(test_input_data[i+10])

print(len(input_matrix))
print(len(true_output_list))
print(len(test_input_matrix))
print(len(test_true_output_list))
print(len(input_data))
