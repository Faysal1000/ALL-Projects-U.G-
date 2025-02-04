namespace VisaProcessing.Migrations
{
    using System;
    using System.Data.Entity.Migrations;
    
    public partial class tokenDateColumnRemove : DbMigration
    {
        public override void Up()
        {
            DropColumn("dbo.Tokens", "TokenDate");
        }
        
        public override void Down()
        {
            AddColumn("dbo.Tokens", "TokenDate", c => c.DateTime(nullable: false));
        }
    }
}
