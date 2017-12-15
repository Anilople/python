#coding=utf-8

import MySQLdb

conn = MySQLdb.connect(
    host='localhost',
    port=3306,
    user='root',
    db='samp_db'
)


cur = conn.cursor()
print cur.execute("show databases;")
cur.execute("use samp_db")
# cur.execute("create table student(id int,name varchar(20),class varchar(30),age varchar(10))")
cur.execute("insert into student values('2','Tom','3 year 2 class','9')")
print cur.execute("show tables")

conn.commit()
cur.close()
conn.close()
