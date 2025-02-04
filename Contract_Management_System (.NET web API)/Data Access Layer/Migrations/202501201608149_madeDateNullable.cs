namespace Data_Access_Layer.Migrations
{
    using System;
    using System.Data.Entity.Migrations;
    
    public partial class madeDateNullable : DbMigration
    {
        public override void Up()
        {
            AlterColumn("dbo.Contacts", "Birthday", c => c.DateTime());
        }
        
        public override void Down()
        {
            AlterColumn("dbo.Contacts", "Birthday", c => c.DateTime(nullable: false));
        }
    }
}
